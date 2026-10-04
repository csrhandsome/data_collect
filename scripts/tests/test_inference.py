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

    def infer(self, obs):
        self.calls += 1
        time.sleep(0.1)
        assert "observation/ee_pose" in obs
        pose = obs["observation/ee_pose"].copy()
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
            return {"actions": np.tile(np.r_[obs["observation/ee_pose"], 1.0], (16, 1))}

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
