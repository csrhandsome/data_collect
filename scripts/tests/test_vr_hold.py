from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control._panda.fake import FakeBackend
from control.collection import pipeline
from control.collection.devices import FakeVR
from control.collection.vr import EEMapper, VRResampler
from control.config import load_config


def measured_pose(z, angle=0.0):
    return SimpleNamespace(
        ee_position=np.array([0.4, 0.0, z]),
        ee_quaternion_xyzw=Rotation.from_rotvec([0.0, angle, 0.0]).as_quat(),
    )


def vr_sample(enabled, x=0.0):
    return SimpleNamespace(
        enabled=enabled,
        position=np.array([x, 0.0, 0.0]),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )


def test_idle_target_stays_fixed_when_measured_pose_drifts():
    mapper = EEMapper({})
    initial = measured_pose(0.4)
    position, quaternion = mapper.map(vr_sample(False), initial, 1_000_000_000)
    # The caller may modify returned arrays without changing the held pose.
    position[:] = 0.0
    quaternion[:] = 0.0
    for tick in range(1, 6):
        state = measured_pose(0.4 - tick * 0.001, tick * 0.01)
        position, quaternion = mapper.map(
            vr_sample(False), state, 1_000_000_000 + tick * 10_000_000
        )
        np.testing.assert_allclose(position, initial.ee_position)
        np.testing.assert_allclose(quaternion, initial.ee_quaternion_xyzw)


def test_trigger_release_latches_measured_pose_once_and_reanchors_on_resume():
    mapper = EEMapper({})
    mapper.map(vr_sample(True), measured_pose(0.4), 1_000_000_000)
    mapper.map(vr_sample(True, x=0.01), measured_pose(0.4), 1_010_000_000)
    released = measured_pose(0.405, 0.02)
    for tick, state in enumerate(
        [released, measured_pose(0.404, 0.03), measured_pose(0.403, 0.04)]
    ):
        position, quaternion = mapper.map(
            vr_sample(False), state, 1_020_000_000 + tick * 10_000_000
        )
        np.testing.assert_allclose(position, released.ee_position)
        np.testing.assert_allclose(quaternion, released.ee_quaternion_xyzw)

    resumed = measured_pose(0.403, 0.04)
    position, quaternion = mapper.map(vr_sample(True, x=0.5), resumed, 1_050_000_000)
    np.testing.assert_allclose(position, resumed.ee_position)
    np.testing.assert_allclose(quaternion, resumed.ee_quaternion_xyzw)
    position, _ = mapper.map(vr_sample(True, x=0.501), resumed, 1_060_000_000)
    np.testing.assert_allclose(position, resumed.ee_position + [0.001, 0.0, 0.0])


def test_stale_vr_latches_hold_instead_of_following_drift():
    mapper, resampler = EEMapper({}), VRResampler(0.5)
    raw = FakeVR().latest
    raw.pose_monotonic_ns = 1_000_000_000
    raw.pos_x = 0.0
    mapper.map(resampler.sample(raw, 1_000_000_000), measured_pose(0.4), 1_000_000_000)
    stopped = measured_pose(0.399)
    mapper.map(resampler.sample(raw, 1_500_000_000), stopped, 1_500_000_000)
    position, quaternion = mapper.map(
        resampler.sample(raw, 1_510_000_000), measured_pose(0.398, 0.01), 1_510_000_000
    )
    np.testing.assert_allclose(position, stopped.ee_position)
    np.testing.assert_allclose(quaternion, stopped.ee_quaternion_xyzw)


def test_startup_and_robot_reset_seed_new_hold_pose():
    mapper = EEMapper({})
    mapper.map(vr_sample(True), measured_pose(0.5), 1_000_000_000)
    for tick, state in enumerate([measured_pose(0.4), measured_pose(0.45, 0.1)]):
        mapper.reset(state)
        expected_position = state.ee_position.copy()
        expected_quaternion = state.ee_quaternion_xyzw.copy()
        state.ee_position[:] = 0.0
        state.ee_quaternion_xyzw[:] = 0.0
        position, quaternion = mapper.map(
            vr_sample(False), measured_pose(0.39), 1_010_000_000 + tick * 10_000_000
        )
        np.testing.assert_allclose(position, expected_position)
        np.testing.assert_allclose(quaternion, expected_quaternion)


@pytest.mark.parametrize("hz", [30, 100])
def test_target_advances_at_configured_speed_when_robot_feedback_lags(hz):
    mapper = EEMapper({"rotation_gain": 1.0})
    state = measured_pose(0.4)
    sample = vr_sample(True)
    previous_position, previous_quaternion = mapper.map(sample, state, 1_000_000_000)
    sample.position[0] = 0.1
    sample.quaternion_xyzw = Rotation.from_rotvec([0.0, 0.4, 0.0]).as_quat()
    for tick in range(1, hz + 1):
        position, quaternion = mapper.map(sample, state, 1_000_000_000 + round(tick * 1e9 / hz))
        assert np.linalg.norm(position - previous_position) <= 0.2 / hz + 1e-9
        angle = (
            Rotation.from_quat(quaternion) * Rotation.from_quat(previous_quaternion).inv()
        ).magnitude()
        assert angle <= 0.2 / hz + 1e-9
        previous_position, previous_quaternion = position, quaternion
    # Repeated VR samples and delayed feedback must not pin the target to the measured pose.
    np.testing.assert_allclose(position, [0.5, 0.0, 0.4], atol=1e-9)
    assert Rotation.from_quat(quaternion).magnitude() == pytest.approx(0.2)


def test_collection_sends_fixed_idle_target_without_starting_recording(tmp_path, monkeypatch):
    class DriftingBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.targets = []
            self.first_stream_position = None

        def snapshot(self, busy=False):
            if self.streaming:
                self.position[2] -= 0.0001
                if self.first_stream_position is None:
                    self.first_stream_position = self.position.copy()
            return super().snapshot(busy)

        def target(self, position, quaternion, nullspace):
            # Inject measured drift independently of the commanded target.
            self.targets.append(np.array(position, copy=True))

    class IdleVR(FakeVR):
        @property
        def latest(self):
            sample = super().latest
            sample.arm_enabled = False
            return sample

    backend = DriftingBackend()
    recorder = Mock(active=False)
    arm_factory = pipeline.RoboticArmControler
    monkeypatch.setattr(
        pipeline,
        "RoboticArmControler",
        lambda **kwargs: arm_factory(config=kwargs["config"], backend=backend),
    )
    monkeypatch.setattr(pipeline, "make_vr", lambda config, dry_run: IdleVR())
    monkeypatch.setattr(pipeline, "open_dataset", lambda config: (Mock(), tmp_path))
    monkeypatch.setattr(pipeline, "EpisodeRecorder", lambda *args: recorder)
    config = load_config()
    config["robot"]["move_to_start"] = False
    config["audio"].update(enabled=False, vad_enabled=False)

    result = pipeline.run_collection(config, dry_run=True, max_steps=5)

    assert result["ticks"] == 5
    # Cleanup explicitly captures the final measured pose; inspect servo ticks only.
    assert len(backend.targets) >= 5
    for target in backend.targets[:5]:
        np.testing.assert_allclose(target, backend.first_stream_position)
    recorder.start.assert_not_called()
