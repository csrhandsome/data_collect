"""Configuration contracts and opt-in real DDS command/feedback integration."""

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from control.base_runner import BaseRunner
from control.config import load_config, validate
from control.runners import create_runner

ROOT = Path(__file__).resolve().parents[3]


def load_backend_config(tmp_path, backend="ros2", purpose="train"):
    path = tmp_path / f"{backend}_{purpose}.yaml"
    parent = ROOT / f"config/{purpose}/panda.yaml"
    path.write_text(f"extends: {parent}\nrobot:\n  backend: {backend}\n")
    return load_config(path)


@pytest.mark.parametrize("purpose", ["train", "inference"])
@pytest.mark.parametrize("backend", ["panda", "ros2"])
def test_shared_config_selects_backend_and_defaults(tmp_path, backend, purpose):
    config = load_backend_config(tmp_path, backend, purpose)
    arm = create_runner(config=config)
    is_ros = backend == "ros2"
    assert isinstance(arm, BaseRunner)
    assert type(arm).__name__ == ("FrankaRunner" if is_ros else "FrankaRunnerStable")
    assert type(arm._backend).__name__ == ("Ros2Backend" if is_ros else "PandaBackend")
    assert arm.get_capabilities().finite_motion is not is_ros
    assert not arm.get_status().connected
    assert config["robot"]["move_to_start"] is not is_ros
    assert config["robot"]["activate_fci"] is not is_ros
    assert config["ros2"]["enabled"] is is_ros
    original = load_config(ROOT / f"config/{purpose}/panda.yaml")
    assert config["camera"]["backend"] == ("ros2" if is_ros else "direct")
    assert config["audio"]["backend"] == ("ros2" if is_ros else "direct")
    for section in ("dataset", "observation", "action"):
        assert config[section] == original[section]
    assert {k: v for k, v in config["camera"].items() if k != "backend"} == {
        k: v for k, v in original["camera"].items() if k != "backend"
    }
    arm.close()


def test_backend_defaults_preserve_explicit_overrides(tmp_path):
    parent = ROOT / "config/train/panda.yaml"
    path = tmp_path / "custom.yaml"
    path.write_text(
        f"extends: {parent}\nrobot:\n  activate_fci: false\n  move_to_start: false\n"
        "ros2:\n  enabled: false\n"
    )
    config = load_config(path)
    assert config["robot"]["activate_fci"] is False
    assert config["robot"]["move_to_start"] is False
    assert config["ros2"]["enabled"] is False
    path.write_text(path.read_text().replace("robot:\n", "robot:\n  backend: ros2\n"))
    assert load_config(path)["ros2"]["enabled"] is False


def test_runner_factory_rejects_unknown_backend_and_abstract_base():
    with pytest.raises(ValueError, match="robot.backend"):
        create_runner(config={"robot": {"backend": "unknown"}})
    with pytest.raises(TypeError):
        BaseRunner()


def test_runner_construction_does_not_import_hardware_sdks():
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'panda_py', 'rclpy', 'sounddevice', 'pyrealsense2'}:
        raise AssertionError('Construction imported hardware SDK: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from control.base_runner import BaseRunner
from control.runners import create_runner
from control.robotic_arm_controller import RoboticArmControler
assert RoboticArmControler is create_runner
for backend in ('panda', 'ros2'):
    runner = create_runner(config={'robot': {'backend': backend}})
    assert isinstance(runner, BaseRunner)
    assert not runner.get_status().connected
    runner.close()
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


def test_ros_robot_can_keep_direct_camera_and_microphone(tmp_path):
    parent = ROOT / "config/train/panda.yaml"
    path = tmp_path / "mixed.yaml"
    path.write_text(
        f"extends: {parent}\nrobot:\n  backend: ros2\n"
        "camera:\n  backend: direct\naudio:\n  backend: direct\n"
    )
    config = load_config(path)
    assert config["camera"]["backend"] == config["audio"]["backend"] == "direct"
    assert type(create_runner(config=config)).__name__ == "FrankaRunner"


@pytest.fixture
def ros_config(tmp_path):
    return load_backend_config(tmp_path)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("robot", "backend", "unknown"),
        ("robot", "move_to_start", True),
        ("robot", "activate_fci", True),
        ("ros2", "state_timeout_s", 0),
        ("ros2", "state_timeout_s", float("nan")),
        ("ros2", "arm_id", ""),
        ("ros2", "base_frame", ""),
        ("inference", "move_to_start", True),
        ("inference", "action_space", "joint"),
    ],
)
def test_ros_config_rejects_incompatible_control(ros_config, section, key, value):
    config = ros_config
    config.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        validate(config)


def test_preflight_detects_upstream_demo_and_accepts_external_only(tmp_path):
    spec = importlib.util.spec_from_file_location("preflight", ROOT / "src/env/check_controller.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "controller.cpp"
    source.write_text(""""~/equilibrium_pose"
CartesianImpedanceExampleController::update(const rclcpp::Time&) {
  updateMotionTarget(period.seconds(), orientation);
}
void CartesianImpedanceExampleController::updateMotionTarget() {}
""")
    with pytest.raises(RuntimeError, match="overwrites"):
        module.check(source)
    source.write_text(
        source.read_text().replace("  updateMotionTarget(period.seconds(), orientation);", "")
    )
    module.check(source)


ROS_TEST = pytest.mark.skipif(
    os.environ.get("DATA_COLLECT_TEST_ROS2") != "1",
    reason="Requires sourced ROS 2 + franka_msgs and DDS sockets",
)


@pytest.fixture
def ros_peer():
    import rclpy
    from franka_msgs.action import Grasp, Move
    from franka_msgs.msg import FrankaRobotState
    from geometry_msgs.msg import PoseStamped
    from rclpy.action import ActionServer
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from std_srvs.srv import Trigger

    context = Context()
    rclpy.init(args=[], context=context)
    namespace = "/test_franka_" + uuid4().hex[:10]
    node = rclpy.create_node("franka_test_peer", namespace=namespace, context=context)
    peer = type("Peer", (), {})()
    peer.namespace, peer.node = namespace, node
    peer.commands, peer.grasps, peer.moves = [], [], []
    peer.publish_state = True
    peer.frame = "panda_link0"
    peer.age_s = 0
    peer.position = [0.4, 0.0, 0.4]
    publisher = node.create_publisher(
        FrankaRobotState, "franka_robot_state_broadcaster/robot_state", 1
    )

    def state():
        if not peer.publish_state:
            return
        msg = FrankaRobotState()
        msg.header.frame_id = peer.frame
        stamp = node.get_clock().now().nanoseconds - round(peer.age_s * 1e9)
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        transform = np.eye(4)
        transform[:3, 3] = peer.position
        msg.o_t_ee = transform.flatten(order="F").tolist()
        msg.q = [i * 0.01 for i in range(1, 8)]
        msg.dq = [0.0] * 7
        publisher.publish(msg)

    def command(msg):
        peer.commands.append(msg)

    node.create_subscription(
        PoseStamped, "cartesian_pose_impedance_controller/equilibrium_pose", command, 1
    )
    node.create_timer(0.01, state)

    def execute_move(handle):
        peer.moves.append(handle.request)
        handle.succeed()
        return Move.Result(success=True)

    def execute_grasp(handle):
        peer.grasps.append(handle.request)
        handle.succeed()
        return Grasp.Result(success=True)

    actions = [
        ActionServer(node, Move, "panda_gripper/move", execute_move),
        ActionServer(node, Grasp, "panda_gripper/grasp", execute_grasp),
    ]
    node.create_service(
        Trigger, "panda_gripper/stop", lambda request, response: Trigger.Response(success=True)
    )
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        yield peer
    finally:
        executor.shutdown(timeout_sec=2)
        thread.join(timeout=2)
        for action in actions:
            action.destroy()
        node.destroy_node()
        context.try_shutdown()
        context.destroy()


def peer_config(peer, config):
    config["ros2"].update(
        namespace=peer.namespace, connect_timeout_s=2, state_timeout_s=0.15, enabled=False
    )
    config["gripper"]["type"] = "none"
    config["reactive_desk"]["reactive_desk_enabled"] = False
    return config


def until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("ROS peer did not observe the expected event")
        time.sleep(0.01)


@ROS_TEST
def test_real_pose_command_state_order_and_stale_guard(ros_peer, ros_config):
    with create_runner(config=peer_config(ros_peer, ros_config)) as arm:
        np.testing.assert_allclose(arm.get_state().joint_positions, np.arange(1, 8) * 0.01)
        assert arm.get_state().base_frame == "panda_link0"
        until(lambda: arm._backend._publisher.get_subscription_count() > 0)
        arm.start_stream()
        arm.send_ee_target([0.41, 0.0, 0.4], [0, 0, 0, 2])
        until(lambda: any(abs(p.pose.position.x - 0.41) < 1e-8 for p in ros_peer.commands))
        command = next(p for p in ros_peer.commands if abs(p.pose.position.x - 0.41) < 1e-8)
        assert command.header.frame_id == "panda_link0"
        assert command.header.stamp.sec > 0
        assert command.pose.orientation.w == 1.0
        with pytest.raises(ValueError, match="translation guard"):
            arm.send_ee_target([2, 0, 0.4], [0, 0, 0, 1])
        ros_peer.publish_state = False
        time.sleep(0.2)
        with pytest.raises(TimeoutError, match="stale"):
            arm.send_ee_target([0.41, 0.0, 0.4], [0, 0, 0, 1])
        ros_peer.publish_state = True
        until(lambda: time.monotonic_ns() - arm._backend._state.sampled_monotonic_ns < 0.1e9)
        arm.stop_stream()


@ROS_TEST
@pytest.mark.parametrize("invalid", ["frame", "stamp"])
def test_invalid_feedback_does_not_refresh_state(ros_peer, ros_config, invalid):
    arm = create_runner(config=peer_config(ros_peer, ros_config))
    arm.connect()
    try:
        if invalid == "frame":
            ros_peer.frame = "wrong_link"
        else:
            ros_peer.age_s = 10
        time.sleep(0.25)
        with pytest.raises(TimeoutError):
            arm.get_state()
    finally:
        with pytest.raises(RuntimeError, match="cleanup"):
            arm.close()
        assert not arm.get_status().connected


@ROS_TEST
def test_real_franka_gripper_actions(ros_peer, ros_config):
    config = peer_config(ros_peer, ros_config)
    config["gripper"]["type"] = "franka"
    with create_runner(config=config) as arm:
        arm.gripper.set_open_ratio(0.8)
        assert ros_peer.moves[0].width == pytest.approx(config["gripper"]["open_width_m"] * 0.8)
        arm.gripper.close()
        assert ros_peer.grasps[0].force == 60
        assert arm.get_state().gripper.commanded_open_ratio == 0
        arm.gripper.stop()


@ROS_TEST
def test_collection_uses_ros_backend_and_preserves_recording(ros_peer, ros_config, tmp_path, monkeypatch):
    from control.collection import pipeline
    from control.collection.devices import FakeCameras, FakeVR

    config = peer_config(ros_peer, ros_config)
    config["dataset"].update(root=str(tmp_path), date="dds", enable_logging=True)
    monkeypatch.setattr(pipeline, "make_cameras", lambda config, dry_run: FakeCameras(config))
    monkeypatch.setattr(pipeline, "make_vr", lambda config, dry_run: FakeVR())
    stats = pipeline.run_collection(config, max_steps=35)
    assert stats["ticks"] == 35
    assert len(ros_peer.commands) > 5
    assert list(tmp_path.rglob("episode_*.sync.json"))


def test_ros_style_replay_requires_initial_alignment_and_skips_finite_moves():
    from control.direct.fake import FakeBackend
    from replay.scripts.replay_robot import replay_trajectory

    backend = FakeBackend()
    backend.finite_motion = False
    config = {"gripper": {"type": "none"}}
    with create_runner(config=config, backend=backend) as arm:
        points = [
            {
                "time_s": 0,
                "position": np.array([0.4, 0, 0.4]),
                "quaternion_xyzw": np.array([0, 0, 0, 1]),
                "joints": np.ones(7),
            },
            {
                "time_s": 0.01,
                "position": np.array([0.401, 0, 0.4]),
                "quaternion_xyzw": np.array([0, 0, 0, 1]),
            },
        ]
        replay_trajectory(arm, points)
        assert "move_joints" not in backend.calls and "move_ee" not in backend.calls
        backend.position = np.array([0.6, 0, 0.4])
        previous = len(backend.calls)
        with pytest.raises(ValueError, match="起始末端位姿"):
            replay_trajectory(arm, points)
        assert len(backend.calls) == previous
