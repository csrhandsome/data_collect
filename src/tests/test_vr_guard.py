"""A lagging robot must reject distant VR goals without losing the runner."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control.collection import pipeline
from control.collection.vr import EEMapper, VRResampler
from control.config import DEFAULT_CONFIG, INFERENCE_CONFIG, load_config
from control.direct.fake import FakeBackend
from control.inference import loop
from control.types import EETargetLimitError
from control.vr_input import VRInput


def test_guard_requires_fresh_release_and_preserves_hold_across_stale_input():
    state = FakeBackend().snapshot()
    mapper, resampler = EEMapper({}), VRResampler()
    raw = VRInput(arm_enabled=True, pose_monotonic_ns=1_000_000_000)
    mapper.suspend_until_release(state)
    mapper.reset()  # Workflow resets must not bypass the pending release.
    for now, enabled, source in [
        (1_010_000_000, True, 1_010_000_000),
        (2_000_000_000, False, 1_000_000_000),  # stale disabled is not a release
        (2_010_000_000, True, 2_010_000_000),  # reconnect while held
    ]:
        raw.arm_enabled, raw.pose_monotonic_ns, raw.pos_x = enabled, source, 10.0
        position, _ = mapper.map(resampler.sample(raw, now), state, now)
        np.testing.assert_allclose(position, state.ee_position)
        assert mapper.waiting_for_release

    raw.arm_enabled, raw.pose_monotonic_ns = False, 2_020_000_000
    mapper.map(resampler.sample(raw, raw.pose_monotonic_ns), state, raw.pose_monotonic_ns)
    assert not mapper.waiting_for_release
    raw.arm_enabled, raw.pose_monotonic_ns = True, 2_030_000_000
    position, _ = mapper.map(resampler.sample(raw, raw.pose_monotonic_ns), state, raw.pose_monotonic_ns)
    np.testing.assert_allclose(position, state.ee_position)  # new anchor, no catch-up jump
    raw.pos_x, raw.pose_monotonic_ns = 10.001, 2_040_000_000
    position, _ = mapper.map(resampler.sample(raw, raw.pose_monotonic_ns), state, raw.pose_monotonic_ns)
    np.testing.assert_allclose(position, state.ee_position + [0.001, 0, 0])


def scripted_run(monkeypatch, module, quantity, *, inference=False):
    clock = SimpleNamespace(tick=0, now=1_000_000_000)

    class Rate:
        overruns = 0

        def __init__(self, hz):
            pass

        def tick(self):
            clock.tick += 1
            clock.now += 10_000_000
            return clock.now

    class LaggingBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.accepted = []

        def target(self, position, quaternion, nullspace):
            # Feedback deliberately never catches up with the advancing VR goal.
            self.accepted.append((clock.tick, np.array(position), np.array(quaternion)))

    class VR:
        def read(self):
            tick = clock.tick
            release = 12 if inference else 10
            x = 0.0 if tick == 1 else (0.5 if tick < release else 5.0)
            if tick > release + 1:
                x += 0.001
            quaternion = Rotation.from_rotvec([0, x, 0]).as_quat()
            return VRInput(
                arm_enabled=(tick < release or tick >= release + 3) if inference else tick != release,
                pos_x=x if quantity == "translation" else 0.0,
                quat_x=quaternion[0] if quantity == "rotation" else 0.0,
                quat_y=quaternion[1] if quantity == "rotation" else 0.0,
                quat_z=quaternion[2] if quantity == "rotation" else 0.0,
                quat_w=quaternion[3] if quantity == "rotation" else 1.0,
                # A gap during a guarded inference takeover must not resume the policy.
                pose_monotonic_ns=clock.now - (1_000_000_000 if inference and tick == 10 else 0),
                pose_seq=tick,
            )

        def close(self):
            pass

    backend = LaggingBackend()
    monkeypatch.setattr(module, "FixedRate", Rate)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic_ns=lambda: clock.now))
    monkeypatch.setattr(module, "make_vr", lambda *args: VR())
    monkeypatch.setattr("control.direct.fake.FakeBackend", lambda: backend)
    config = load_config(INFERENCE_CONFIG if inference else DEFAULT_CONFIG)
    config["robot"]["move_to_start"] = False
    config["ros2"]["enabled"] = False
    config["dataset"]["enable_logging"] = False
    config["audio"].update(enabled=False, vad_enabled=False)
    config["control"].update(max_target_translation_m=0.015, max_target_rotation_rad=0.05)
    config["vr"].update(rotation_gain=1.0, max_rotation_speed_rad_s=2.0)
    return backend, config


@pytest.mark.parametrize("quantity", ["translation", "rotation"])
def test_collection_guard_holds_rearms_and_keeps_runner_alive(monkeypatch, caplog, quantity):
    backend, config = scripted_run(monkeypatch, pipeline, quantity)
    result = pipeline.run_collection(config, dry_run=True, max_steps=12)
    assert result["ticks"] == 12
    assert caplog.text.count(f"exceeds measured {quantity} guard") == 1
    assert "target_xyz=" in caplog.text and "measured_xyz=" in caplog.text
    for tick, position, quaternion in backend.accepted:
        assert np.linalg.norm(position - backend.position) <= 0.015 + 1e-12
        assert Rotation.from_quat(quaternion).magnitude() <= 0.05 + 1e-12
        if 6 <= tick <= 11:
            np.testing.assert_allclose(position, backend.position)
            np.testing.assert_allclose(quaternion, backend.quaternion)
    # Motion resumes from a new anchor, even though the operator moved while blocked.
    _, position, quaternion = next(t for t in backend.accepted if t[0] == 12)
    if quantity == "translation":
        np.testing.assert_allclose(position, backend.position + [0.001, 0, 0])
    else:
        assert Rotation.from_quat(quaternion).magnitude() == pytest.approx(0.001)


def test_other_backend_value_errors_still_fail_collection(monkeypatch):
    backend, config = scripted_run(monkeypatch, pipeline, "translation")
    target = backend.target

    def reject_once(*args):
        backend.target = target
        raise ValueError("backend failure")

    backend.target = reject_once
    with pytest.raises(ValueError, match="backend failure"):
        pipeline.run_collection(config, dry_run=True, max_steps=12)


def test_public_guard_rejects_target_and_reports_all_xyz():
    from control.runners import create_runner

    backend = FakeBackend()
    with create_runner(backend=backend) as arm:
        arm.start_stream()
        before = arm.get_ee_target()
        with pytest.raises(EETargetLimitError) as caught:
            arm.send_ee_target([0.56, 0.1, 0.41], [0, 0, 0, 1])
        assert caught.value.details["error"] == pytest.approx(np.linalg.norm([0.16, 0.1, 0.01]))
        assert caught.value.details["limit"] == 0.15
        assert caught.value.details["measured_position"] == [0.4, 0.0, 0.4]
        assert arm.get_ee_target().submitted_monotonic_ns == before.submitted_monotonic_ns
        assert "target" not in backend.calls


def test_guarded_inference_does_not_resume_policy_on_stale_vr(monkeypatch, caplog):
    from unittest.mock import Mock

    backend, config = scripted_run(monkeypatch, loop, "translation", inference=True)
    config["inference"]["enable_vr_takeover"] = True
    recorder = Mock(index=0)
    recorder.dataset.meta.total_episodes = 0
    monkeypatch.setattr(loop, "open_dataset", lambda config: (Mock(), None))
    monkeypatch.setattr(loop, "EpisodeRecorder", lambda *args: recorder)

    class Policy:
        server_metadata = {
            "action_space": "ee", "action_fps": 30, "action_horizon": 50, "action_dim": 7,
        }

        def infer(self, observation):
            return {"actions": np.tile(np.r_[observation["observation.ee_pose"], 1.0], (50, 1))}

        def close(self):
            pass

    result = loop.run_inference(config, dry_run=True, max_steps=14, policy=Policy())
    assert result["ticks"] == 14
    assert result["requests"] == 1  # only after the fresh release at tick 12
    assert caplog.text.count("exceeds measured translation guard") == 1
    assert [c.args[0] for c in recorder.control_event.call_args_list] == ["takeover_start", "takeover_end"]
    for tick, position, _ in backend.accepted:
        if 6 <= tick <= 12:
            np.testing.assert_allclose(position, backend.position)
