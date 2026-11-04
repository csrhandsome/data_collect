"""UR kinematics, command bounds and six-axis recording without ROS or hardware."""

import copy
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from control.collection.dataset import open_dataset
from control.collection.devices import FakeVR
from control.collection.recording import EpisodeRecorder
from control.collection.vr import VRResampler
from control.config import load_config, validate
from control.direct.fake import FakeBackend
from control.ros2.ur_config import UR_JOINT_NAMES
from control.ros2.ur_kinematics import URKinematics
from control.ros2.ur_runner import URRos2Backend, URRunner
from control.runners import create_runner
from control.types import CameraPair
from control.util.pose import matrix_to_quat_xyzw
from env.bringup_ur import launch_command

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def config():
    return load_config(ROOT / "config/train/ur.yaml")


@pytest.fixture
def model():
    joints = []
    for i, (name, axis) in enumerate(zip(UR_JOINT_NAMES, ("0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "1 0 0"), strict=True)):
        joints.append(f'''<joint name="{name}" type="revolute">
          <parent link="link{i}"/><child link="link{i + 1}"/>
          <origin xyz="0.1 0 0.1" rpy="0 0 0"/><axis xyz="{axis}"/>
          <limit lower="-3" upper="3" velocity="1" effort="10"/>
        </joint>''')
    joints.append('<joint name="tool" type="fixed"><parent link="link6"/><child link="tool0"/><origin xyz="0.1 0 0" rpy="0 0 1.57"/></joint>')
    return URKinematics("<robot>" + "".join(joints) + "</robot>", "link0", "tool0", UR_JOINT_NAMES)


def test_ur_config_and_factory_keep_franka_default(config):
    assert isinstance(create_runner(config=config), URRunner)
    assert create_runner()._backend.__class__.__name__ == "PandaBackend"
    assert load_config()["robot"]["backend"] == "ros2"
    assert config["camera"]["backend"] == config["audio"]["backend"] == "ros2"
    assert not config["robot"]["move_to_start"]
    from control.ros2.diagnostics_bridge import expected_topics

    topics = expected_topics(config)
    assert "/joint_states" in topics and "/robot_description" in topics and "/vr/input" in topics
    assert not any("franka" in topic or "panda" in topic for topic in topics)
    with create_runner(config=config, backend=FakeBackend(6)) as arm:
        assert arm.get_state().joint_positions.shape == (6,)
        assert not arm.get_capabilities().gripper
        arm.move_joints(np.zeros(6))
        with pytest.raises(ValueError):
            arm.move_joints(np.zeros(7))
        arm.start_stream()
        arm.send_ee_target([0.401, 0, 0.4], [0, 0, 0, 1])
        arm.hold()
        arm.stop_stream()


@pytest.mark.parametrize("section,key,value", [
    ("robot", "start_joint_position", [0] * 7),
    ("robot", "activate_fci", True),
    ("gripper", "type", "franka"),
    ("ros2", "joint_names", ["duplicate"] * 6),
    ("ros2", "max_joint_velocity_rad_s", float("nan")),
    ("ros2", "command_duration_s", 1.0),
    ("ros2", "max_command_joint_error_rad", 0.3),
    ("ros2", "use_mock_hardware", "false"),
])
def test_reject_invalid_ur_config(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate(config)


def test_calibrated_chain_jacobian_and_local_inverse(model):
    seed = np.array([0.2, -0.5, 0.8, -0.3, 0.5, -0.2])
    pose, jacobian = model.forward(seed, jacobian=True)
    numeric = np.zeros((6, 6))
    for axis in range(6):
        perturbed = seed.copy()
        perturbed[axis] += 1e-7
        moved = model.forward(perturbed)
        numeric[:, axis] = np.r_[(moved[:3, 3] - pose[:3, 3]) / 1e-7,
                                Rotation.from_matrix(moved[:3, :3] @ pose[:3, :3].T).as_rotvec() / 1e-7]
    np.testing.assert_allclose(jacobian, numeric, atol=1e-6)
    desired = model.forward(seed + 0.01)
    solved = model.inverse(desired[:3, 3], matrix_to_quat_xyzw(desired[:3, :3]), seed)
    np.testing.assert_allclose(model.forward(solved), desired, atol=1e-4)
    with pytest.raises(ValueError, match="limits"):
        model.validate(np.ones(6) * 4)
    with pytest.raises(ValueError, match="converge"):
        model.inverse([10, 0, 0], [0, 0, 0, 1], seed)


@pytest.fixture
def backend(config, model):
    backend = URRos2Backend(config)
    backend._model = model
    stamp = SimpleNamespace(nanoseconds=time.time_ns())
    backend._source.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: stamp),
        get_logger=lambda: SimpleNamespace(warning=lambda *_, **__: None),
    )
    backend.gripper = SimpleNamespace(get_state=lambda **kw: FakeBackend(6).snapshot().gripper)
    messages = []
    backend._publisher = SimpleNamespace(publish=messages.append, get_subscription_count=lambda: 1)
    backend._trajectory_type = lambda: SimpleNamespace()
    backend._point_type = lambda: SimpleNamespace(time_from_start=SimpleNamespace(sec=0, nanosec=0))
    joints = np.array([0.2, -0.5, 0.8, -0.3, 0.5, -0.2])
    # Deliberately shuffled incoming joint order.
    msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(
        sec=stamp.nanoseconds // 10**9, nanosec=stamp.nanoseconds % 10**9)),
        name=list(reversed(UR_JOINT_NAMES)), position=joints[::-1], velocity=np.zeros(6))
    backend._on_state(msg)
    return backend, messages, msg


def test_state_reorders_joints_and_rejects_duplicates_stale_and_nonfinite(backend):
    backend, _, msg = backend
    state = backend.snapshot()
    assert state.joint_positions[0] == pytest.approx(0.2)
    assert backend._on_state(msg) is False
    malformed = copy.deepcopy(msg)
    malformed.header.stamp.nanosec += 1
    malformed.position[0] = np.nan
    assert backend._on_state(malformed) is False
    assert backend.snapshot().sampled_monotonic_ns == state.sampled_monotonic_ns
    backend._state = replace(state, sampled_monotonic_ns=time.monotonic_ns() - 10**9)
    with pytest.raises(TimeoutError, match="stale"):
        backend.snapshot()


def test_bounded_trajectory_hold_and_ik_rejection(backend):
    backend, messages, _ = backend
    state = backend.snapshot()
    with pytest.raises(RuntimeError, match="not started"):
        backend.target(state.ee_position, state.ee_quaternion_xyzw, None)
    backend._streaming = True
    desired = backend._model.forward(state.joint_positions + 0.08)
    backend.target(desired[:3, 3], matrix_to_quat_xyzw(desired[:3, :3]), None)
    message = messages[-1]
    assert message.joint_names == list(UR_JOINT_NAMES)
    delta = np.abs(np.array(message.points[0].positions) - state.joint_positions)
    assert delta.max() <= 0.3 * backend.duration + 1e-9
    assert message.points[0].time_from_start.nanosec == 150_000_000
    count = len(messages)
    with pytest.raises(ValueError):
        backend.target([10, 0, 0], [0, 0, 0, 1], None)
    assert len(messages) == count
    backend.stop()
    np.testing.assert_allclose(messages[-1].points[0].positions, state.joint_positions)
    assert not backend._streaming
    backend._publisher.get_subscription_count = lambda: 0
    with pytest.raises(RuntimeError, match="subscriber"):
        backend._publish_joints(state.joint_positions)


def test_ur_launch_never_uses_franka_desk(config):
    command = launch_command(config, ["use_mock_hardware:=true", "robot_ip:=192.0.2.1"])
    assert command[:4] == ["ros2", "launch", "ur_robot_driver", "ur_control.launch.py"]
    assert "ur_type:=ur5" in command
    assert "use_mock_hardware:=true" in command
    assert "robot_ip:=192.0.2.1" in command
    assert not any("franka" in item or "desk" in item for item in command)
    with pytest.raises(ValueError):
        launch_command(config, ["--bad"])


def test_six_axis_episode_schema_and_saved_values(config, tmp_path):
    config["dataset"].update(root=str(tmp_path), date="ur_test", compact_on_stop=False)
    config["camera"]["image_hw"] = 16
    dataset, root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, root, config)
    fake = FakeBackend(6)
    try:
        recorder.start()
        for index in range(4):
            fake.q[:] = index / 100
            state = fake.snapshot()
            sample = VRResampler().sample(FakeVR().latest, state.sampled_monotonic_ns)
            image = np.full((16, 16, 3), index * 30, dtype=np.uint8)
            recorder.trace(state, sample, state.ee_position, state.ee_quaternion_xyzw)
            recorder.frame(state, CameraPair(image, image, state.sampled_monotonic_ns,
                                            state.sampled_monotonic_ns, 0, 0))
        recorder.finish(fake.snapshot(), publish=True)
    finally:
        recorder.close()
    from replay.scripts._common import _episode_data_path, _load_metadata

    info, episodes = _load_metadata(root)
    assert info["robot_type"] == "ur5"
    assert info["features"]["observation.joint_position"]["shape"] == [6]
    assert info["features"]["action.joint_position"]["shape"] == [6]
    rows = pq.read_table(_episode_data_path(root, info, episodes[0])).to_pylist()
    assert len(rows) == 4
    assert len(rows[0]["action.joint_position"]) == 6
    assert "action.gripper_position" not in rows[0]
