import copy
import time

import numpy as np
import pytest

from control.config import load_config
from control.inference.loop import run_inference
from control.inference.policy import ActionPlan, validate_metadata


class SlowPolicy:
    server_metadata = {
        "action_space": "ee",
        "action_fps": 30.0,
        "action_horizon": 16,
        "action_dim": 7,
    }

    def __init__(self):
        self.closed = False
        self.calls = 0
        self.warmup_finished_at = None
        self.observation_times = []

    def infer(self, obs):
        self.calls += 1
        self.observation_times.append(float(obs["monotonic_ts"]))
        time.sleep(0.1)
        assert "observation.ee_pose" in obs
        pose = obs["observation.ee_pose"].copy()
        pose[0] += 0.001
        if self.calls == 5:
            self.warmup_finished_at = time.monotonic()
        return {"actions": np.tile(np.r_[pose, 1.0], (16, 1))}

    def close(self):
        self.closed = True


def test_async_inference_preserves_servo_rate():
    cfg = copy.deepcopy(load_config())
    cfg["control"]["frequency_hz"] = 100.0
    policy = SlowPolicy()
    stats = run_inference(cfg, dry_run=True, max_steps=60, policy=policy)
    assert stats["ticks"] == 60 and policy.closed
    assert 2 <= stats["requests"] <= 3
    assert (np.diff(policy.observation_times) > 0).all()
    assert time.monotonic() - policy.warmup_finished_at < 1.1


def test_warmup_discards_five_responses_before_starting_stream(monkeypatch):
    from control._panda.fake import FakeBackend

    backend = FakeBackend()
    monkeypatch.setattr("control._panda.fake.FakeBackend", lambda: backend)

    class WarmupPolicy(SlowPolicy):
        def infer(self, obs):
            self.calls += 1
            if self.calls <= 5:
                assert not backend.streaming
                assert "target" not in backend.calls and "gripper" not in backend.calls
                # These actions must never reach the robot.
                return {"actions": np.full((16, 7), 100.0)}
            assert backend.streaming
            return {"actions": np.tile(np.r_[obs["observation.ee_pose"], 1.0], (16, 1))}

    policy = WarmupPolicy()
    stats = run_inference(load_config(), dry_run=True, max_steps=10, policy=policy)
    assert policy.calls == 6 and policy.closed
    assert stats["requests"] == 1 and stats["ticks"] == 10


def test_warmup_failure_does_not_start_stream(monkeypatch):
    from control._panda.fake import FakeBackend

    backend = FakeBackend()
    monkeypatch.setattr("control._panda.fake.FakeBackend", lambda: backend)

    class FailingPolicy(SlowPolicy):
        def infer(self, obs):
            self.calls += 1
            if self.calls == 3:
                raise TimeoutError("warmup timeout")
            return {}

    policy = FailingPolicy()
    with pytest.raises(TimeoutError, match="warmup timeout"):
        run_inference(load_config(), dry_run=True, max_steps=10, policy=policy)
    assert policy.calls == 3 and policy.closed
    assert "start" not in backend.calls and not backend.connected


def test_action_timing_and_metadata_validation():
    plan = ActionPlan(0, np.arange(16 * 7).reshape(16, 7), np.zeros(6))
    assert plan.at(0, 30)[0] is None
    assert plan.at(33_333_334, 30)[1] == 0
    assert plan.at(100_000_000, 30)[1] == 2
    assert plan.at(1_000_000_000, 30)[0] is None
    assert not validate_metadata(SlowPolicy.server_metadata, load_config())
    with pytest.raises(ValueError):
        validate_metadata({**SlowPolicy.server_metadata, "action_fps": 100}, load_config())


@pytest.mark.parametrize("group", ["B", "C"])
def test_force_metadata_and_response_need_no_checksum(group):
    from control._panda.fake import FakeBackend
    from control.inference.policy import parse_response

    cfg = load_config()
    cfg["inference"]["action_space"] = "joint"
    cfg["tactile"]["enabled"] = group == "C"
    metadata = {
        **SlowPolicy.server_metadata,
        "action_space": "joint",
        "action_dim": 7,
        "rlt_protocol": "pytorch-rl-token-v1",
        "rlt_actor_protocol": "pytorch-force-rlt-actor-v1",
        "group": group,
        "finger_input_protocol": "panda-finger-rgb-history-v1",
    }
    assert validate_metadata(metadata, cfg)
    state = FakeBackend().snapshot()
    response = {
        "proposed_q": np.zeros((16, 7)),
        "group": group,
        "ref_obs_ts": state.sampled_monotonic_ns / 1e9,
    }
    assert parse_response(response, state.sampled_monotonic_ns, state, metadata).actions.shape == (
        16,
        7,
    )
    for changed in ({"group": "wrong"}, {"ref_obs_ts": 0}, {"proposed_q": np.zeros((16, 8))}):
        with pytest.raises(ValueError):
            parse_response({**response, **changed}, state.sampled_monotonic_ns, state, metadata)


def test_request_observation_keys_match_recording_schema():
    from control._panda.fake import FakeBackend
    from control.collection.devices import FakeCameras
    from control.collection.schema import features
    from control.inference.policy import build_observation

    cfg = load_config()
    backend = FakeBackend()
    cameras = FakeCameras(cfg["camera"])
    try:
        request = build_observation(
            backend.snapshot(),
            cameras.get_frames(),
            "test",
            "ee",
            tactile=backend.tactile_images(),
            fields=cfg["observation"],
        )
        expected = {
            key
            for key in features(224, "ee", True, cfg["observation"], cfg["action"])
            if key.startswith("observation.")
        }
        assert {key for key in request if key.startswith("observation.")} == expected
    finally:
        cameras.close()


def test_c_slow_policy_does_not_starve_rgb_history_or_servo():
    class CPolicy(SlowPolicy):
        server_metadata = {
            **SlowPolicy.server_metadata,
            "action_space": "joint",
            "action_dim": 7,
            "rlt_protocol": "pytorch-rl-token-v1",
            "rlt_actor_protocol": "pytorch-force-rlt-actor-v1",
            "group": "C",
            "finger_input_protocol": "panda-finger-rgb-history-v1",
            "finger_max_skew_s": 0.02,
            "finger_max_age_s": 0.2,
        }

        def infer(self, obs):
            self.calls += 1
            assert obs["observation.gripper_image_left"].shape == (4, 224, 224, 3)
            assert obs["finger_valid_mask"].all()
            assert (np.diff(obs["finger_timestamps"], axis=0) > 0).all()
            assert "z_grip" not in obs
            time.sleep(0.1)
            if self.calls == 5:
                self.warmup_finished_at = time.monotonic()
            targets = np.r_[obs["observation.ee_pose"][:3], np.zeros(4)]
            return {
                "proposed_q": np.tile(targets, (16, 1)),
                "group": "C",
                "ref_obs_ts": obs["monotonic_ts"],
            }

    cfg = load_config()
    cfg["inference"]["action_space"] = "joint"
    cfg["tactile"]["enabled"] = True
    cfg["gripper"]["type"] = "dh5"
    policy = CPolicy()
    stats = run_inference(cfg, dry_run=True, max_steps=60, policy=policy)
    assert stats["ticks"] == 60 and 2 <= stats["requests"] <= 3
    assert time.monotonic() - policy.warmup_finished_at < 1.1


def test_finger_history_rejects_duplicate_and_stale_frames():
    from types import SimpleNamespace

    from control.inference.fingers import FingerImages

    image = np.zeros((224, 224, 3), np.uint8)
    clock = [1_000_000_000]
    gripper = SimpleNamespace(
        get_tactile_frames=lambda: (image, image, clock[0], clock[0]),
        get_state=lambda: SimpleNamespace(commanded_open_ratio=1.0),
    )
    history = FingerImages(
        load_config(),
        {
            "finger_input_protocol": "panda-finger-rgb-history-v1",
            "finger_max_skew_s": 0.02,
            "finger_max_age_s": 0.2,
        },
    )
    arm = SimpleNamespace(gripper=gripper)
    history.capture(arm)
    history.capture(arm)
    assert len(history.history) == 1
    for _ in range(3):
        clock[0] += 33_333_333
        history.capture(arm)
    assert history.request_fields(clock[0] / 1e9) is not None
    assert history.request_fields(clock[0] / 1e9 + 0.3) is None
