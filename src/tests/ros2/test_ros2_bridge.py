"""Hardware-free bridge contract tests; real ROS pub/sub is tested separately."""

import builtins
import json
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from control.collection.devices import CameraPair
from control.ros2 import bridge as ros2_bridge
from control.ros2.bridge import Ros2Publisher


def header():
    return SimpleNamespace(frame_id="", stamp=SimpleNamespace(sec=0, nanosec=0))


class JointState:
    def __init__(self):
        self.header = header()
        self.name, self.position, self.velocity, self.effort = [], [], [], []


class PoseStamped:
    def __init__(self):
        self.header = header()
        self.pose = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
        )


class Image:
    def __init__(self):
        self.header = header()


class String:
    def __init__(self):
        self.data = ""


@pytest.fixture
def ros(monkeypatch):
    """Small ROS boundary double, including resource ownership and QoS."""
    graph = SimpleNamespace(
        contexts=[],
        nodes=[],
        init_calls=[],
        publishers={},
        condition=threading.Condition(),
        hooks={},
        init_error=None,
        node_error=None,
        publisher_error_at=None,
        destroy_error=False,
        shutdown_error=False,
        system_ns=1_700_000_000_123_456_789,
    )

    class Context:
        def __init__(self):
            self.shutdown_calls = self.destroy_calls = 0
            self.initialized = False
            graph.contexts.append(self)

        def try_shutdown(self):
            self.shutdown_calls += 1
            self.initialized = False
            if graph.shutdown_error:
                raise RuntimeError("context shutdown failed")

        def destroy(self):
            self.destroy_calls += 1

    class Publisher:
        def __init__(self, kind, topic, qos):
            self.kind, self.topic, self.qos = kind, topic, qos
            self.messages, self.thread_ids, self.publish_times = [], [], []

        def publish(self, message):
            hook = graph.hooks.get(self.topic)
            if hook is not None:
                hook(message)
            with graph.condition:
                self.messages.append(message)
                self.thread_ids.append(threading.get_ident())
                self.publish_times.append(time.monotonic())
                graph.condition.notify_all()

    class Node:
        def __init__(self):
            self.destroy_calls = 0
            graph.nodes.append(self)

        def create_publisher(self, kind, topic, qos):
            if len(graph.publishers) == graph.publisher_error_at:
                raise RuntimeError("publisher creation failed")
            publisher = Publisher(kind, topic, qos)
            graph.publishers[topic] = publisher
            return publisher

        def destroy_node(self):
            self.destroy_calls += 1
            if graph.destroy_error:
                raise RuntimeError("node destroy failed")

    def init(**kwargs):
        graph.init_calls.append(kwargs)
        # Model a failure after the context has already allocated native state.
        kwargs["context"].initialized = True
        if graph.init_error:
            raise graph.init_error

    def create_node(name, **kwargs):
        graph.node_name, graph.node_kwargs = name, kwargs
        if graph.node_error:
            raise graph.node_error
        return Node()

    class Clock:
        def __init__(self, clock_type):
            assert clock_type == "SYSTEM_TIME"

        def now(self):
            return SimpleNamespace(nanoseconds=graph.system_ns)

    def module(name, **attrs):
        result = ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)

    module("rclpy", init=init, create_node=create_node)
    module("rclpy.context", Context=Context)
    module("rclpy.clock", Clock=Clock, ClockType=SimpleNamespace(SYSTEM_TIME="SYSTEM_TIME"))
    module("rclpy.signals", SignalHandlerOptions=SimpleNamespace(NO="NO"))
    module(
        "rclpy.qos",
        QoSProfile=lambda **kwargs: SimpleNamespace(**kwargs),
        HistoryPolicy=SimpleNamespace(KEEP_LAST="KEEP_LAST"),
        ReliabilityPolicy=SimpleNamespace(RELIABLE="RELIABLE", BEST_EFFORT="BEST_EFFORT"),
        DurabilityPolicy=SimpleNamespace(VOLATILE="VOLATILE"),
    )
    for name, messages in (
        ("geometry_msgs", {"PoseStamped": PoseStamped}),
        ("sensor_msgs", {"Image": Image, "JointState": JointState}),
        ("std_msgs", {"String": String}),
    ):
        module(name)
        module(f"{name}.msg", **messages)

    def wait_for(topic, count=1):
        with graph.condition:
            assert graph.condition.wait_for(
                lambda: len(graph.publishers[topic].messages) >= count, timeout=3.0
            ), f"Timed out waiting for {topic} message {count}"
        return graph.publishers[topic].messages[count - 1]

    graph.wait_for = wait_for
    return graph


def config(**overrides):
    return {"ros2": {"enabled": True, "publish_hz": 1000.0, **overrides}}


def state(ns=8_900_000_123):
    return SimpleNamespace(
        joint_positions=np.arange(7, dtype=float),
        joint_velocities=np.arange(7, dtype=float) / 10,
        ee_position=np.array([0.4, -0.2, 0.3]),
        ee_quaternion_xyzw=np.array([0.0, 0.0, 0.6, 0.8]),
        sampled_monotonic_ns=ns,
        base_frame="panda_link0",
        robot_time_s=17.25,
        gripper=SimpleNamespace(
            kind="franka", commanded_open_ratio=0.7, measured_width_m=0.051, busy=True
        ),
    )


def vr(ns=8_800_000_777):
    return SimpleNamespace(
        pos_x=0.1,
        pos_y=0.2,
        pos_z=0.3,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        quat_w=1.0,
        arm_enabled=True,
        pose_monotonic_ns=ns,
        pose_seq=42,
        timestamp=8.8001,
        x_pressed=True,
        y_pressed=False,
        save_pressed=True,
        gripper_open=False,
        gripper_close=True,
        right_stick_x=-0.2,
        right_stick_y=-0.5,
        gripper_velocity_axis=0.5,
        raw_position=[0.12, 0.21, 0.31],
        raw_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
    )


def cameras(front_ns=8_700_000_111, wrist_ns=8_600_000_222):
    rgb = np.array([[[255, 1, 2], [3, 254, 5]], [[6, 7, 253], [9, 10, 11]]], dtype=np.uint8)
    # Non-contiguous inputs must serialize as compact row-major RGB.
    return CameraPair(rgb[:, ::-1], rgb[::-1], front_ns, wrist_ns, 1234.5, 9876.25)


def stamp_ns(message):
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def test_disabled_never_imports_ros_or_starts_worker():
    # A fresh interpreter proves isolation even if ROS was imported by other tests.
    script = """
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('rclpy', 'sensor_msgs', 'geometry_msgs', 'std_msgs'):
        raise AssertionError('Disabled bridge imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from control.ros2.bridge import Ros2Publisher
for config in ({}, {'ros2': {'enabled': False}}):
    with Ros2Publisher(config) as bridge:
        assert bridge.enabled is False
        assert bridge._thread is None
        bridge.submit(object(), vr=object(), target=object(), cameras=object())
    bridge.close()
assert not any(name.startswith('rclpy') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


def test_topics_clock_provenance_frames_rgb_and_qos(ros, monkeypatch):
    anchor_ns = 9_000_000_000
    monkeypatch.setattr(ros2_bridge.time, "monotonic_ns", lambda: anchor_ns)
    robot, raw, pair = state(), vr(), cameras()
    target = (np.array([0.5, 0.0, 0.6]), np.array([0.0, 0.0, 0.8, 0.6]))
    with Ros2Publisher(config(telemetry_namespace="/lab", node_name="telemetry")) as bridge:
        bridge.submit(robot, vr=raw, target=target, cameras=pair)
        metadata = json.loads(ros.wait_for("timestamps").data)
        assert bridge.enabled is True
        assert len(ros.publishers) == 9
        assert ros.node_name == "telemetry"
        assert ros.node_kwargs["namespace"] == "/lab"
        assert ros.node_kwargs["context"] is ros.contexts[0]
        assert ros.node_kwargs["use_global_arguments"] is False
        assert ros.init_calls[0] == {
            "args": [],
            "context": ros.contexts[0],
            "signal_handler_options": "NO",
        }

        def expected(ns):
            return ros.system_ns + ns - anchor_ns

        joint = ros.wait_for("robot/joint_states")
        pose = ros.wait_for("robot/ee_pose")
        assert joint.name == [f"panda_joint{i}" for i in range(1, 8)]
        assert joint.position == robot.joint_positions.tolist()
        assert joint.velocity == robot.joint_velocities.tolist()
        assert joint.header.frame_id == pose.header.frame_id == "panda_link0"
        assert stamp_ns(joint) == stamp_ns(pose) == expected(robot.sampled_monotonic_ns)
        assert pose.pose.position.x == 0.4 and pose.pose.orientation.z == 0.6
        target_message = ros.wait_for("robot/target_ee_pose")
        assert target_message.header.frame_id == "panda_link0"
        assert target_message.pose.position.x == 0.5
        assert target_message.pose.orientation.z == 0.8
        assert stamp_ns(target_message) == expected(anchor_ns)
        grip = json.loads(ros.wait_for("robot/gripper_state").data)
        assert grip == {
            "kind": "franka",
            "commanded_open_ratio": 0.7,
            "measured_width_m": 0.051,
            "busy": True,
            "sampled_monotonic_ns": robot.sampled_monotonic_ns,
            "frame_id": "panda_link0",
            "stamp": vars(joint.header.stamp),
        }
        vr_pose = ros.wait_for("vr/pose")
        vr_input = json.loads(ros.wait_for("vr/input").data)
        assert vr_pose.header.frame_id == "vr_world"
        assert stamp_ns(vr_pose) == expected(raw.pose_monotonic_ns)
        assert vr_pose.pose.position.x == 0.1
        assert vr_input["position"] == [0.1, 0.2, 0.3]
        assert vr_input["raw_position"] == [0.12, 0.21, 0.31]
        assert vr_input["quaternion_xyzw"] == vr_input["raw_quaternion_xyzw"] == [0, 0, 0, 1]
        assert vr_input["enabled"] is True and vr_input["buttons"]["save_pressed"] is True
        assert vr_input["buttons"]["gripper_close"] is True
        assert vr_input["pose_monotonic_ns"] == raw.pose_monotonic_ns
        assert vr_input["pose_seq"] == 42 and vr_input["timestamp"] == 8.8001
        assert vr_input["gripper_velocity_axis"] == 0.5
        for name, prefix in (("external", "front"), ("wrist", "wrist")):
            message = ros.wait_for(f"camera/{name}/image")
            assert message.header.frame_id == f"{name}_camera_optical_frame"
            assert stamp_ns(message) == expected(getattr(pair, f"{prefix}_ns"))
            assert (message.width, message.height, message.step) == (2, 2, 6)
            assert message.encoding == "rgb8" and message.is_bigendian == 0
            assert message.data == getattr(pair, prefix).tobytes(order="C")
            camera_metadata = metadata["cameras"][name]
            assert camera_metadata["host_capture_monotonic_ns"] == getattr(pair, f"{prefix}_ns")
            assert camera_metadata["sensor_timestamp"] == getattr(
                pair, f"{prefix}_sensor_timestamp"
            )
            assert camera_metadata["image_published"] is True
            assert camera_metadata["stamp"] == vars(message.header.stamp)
        assert metadata["clock"] == {
            "header_clock": "ros_system_time",
            "anchor_monotonic_ns": anchor_ns,
            "anchor_ros_ns": ros.system_ns,
        }
        assert metadata["robot"]["robot_time_s"] == 17.25
        assert metadata["robot"]["sampled_monotonic_ns"] == robot.sampled_monotonic_ns
        assert metadata["robot"]["stamp"] == vars(joint.header.stamp)
        assert metadata["vr"]["pose_seq"] == 42
        assert metadata["target"]["submitted_monotonic_ns"] == anchor_ns
        for topic, publisher in ros.publishers.items():
            assert publisher.qos.depth == 1
            assert publisher.qos.history == "KEEP_LAST"
            assert publisher.qos.durability == "VOLATILE"
            assert publisher.qos.reliability == (
                "BEST_EFFORT" if topic.startswith("camera/") else "RELIABLE"
            )
            assert set(publisher.thread_ids) == {bridge._thread.ident}
    assert ros.nodes[0].destroy_calls == 1
    assert ros.contexts[0].shutdown_calls == ros.contexts[0].destroy_calls == 1
    assert bridge.enabled is False


def test_custom_frames_and_missing_optional_inputs(ros):
    with Ros2Publisher(
        config(
            vr_frame="tracking",
            external_camera_frame="front_optical",
            wrist_camera_frame="hand_optical",
        )
    ) as bridge:
        robot = state()
        robot.base_frame = "custom_base"
        bridge.submit(robot, vr=vr(), cameras=cameras())
        ros.wait_for("timestamps")
        assert ros.wait_for("robot/ee_pose").header.frame_id == "custom_base"
        assert ros.wait_for("vr/pose").header.frame_id == "tracking"
        assert ros.wait_for("camera/external/image").header.frame_id == "front_optical"
        assert ros.wait_for("camera/wrist/image").header.frame_id == "hand_optical"
        bridge.submit(robot)
        metadata = json.loads(ros.wait_for("timestamps", 2).data)
        assert metadata["vr"] is metadata["target"] is None and metadata["cameras"] == {}
        assert len(ros.publishers["vr/pose"].messages) == 1
        assert ros.publishers["robot/target_ee_pose"].messages == []


def test_duplicate_frames_are_independent_and_old_frames_do_not_reappear(ros):
    with Ros2Publisher(config()) as bridge:
        for i, (front, wrist, counts) in enumerate(
            [
                (100, 200, (1, 1)),
                (100, 200, (1, 1)),
                (101, 200, (2, 1)),
                (101, 201, (2, 2)),
                (100, 200, (2, 2)),
            ],
            1,
        ):
            bridge.submit(state(), cameras=cameras(front, wrist))
            metadata = json.loads(ros.wait_for("timestamps", i).data)
            assert (
                tuple(
                    len(ros.publishers[f"camera/{name}/image"].messages)
                    for name in ("external", "wrist")
                )
                == counts
            )
            if i in (2, 5):
                assert not any(camera["image_published"] for camera in metadata["cameras"].values())
        assert len(ros.publishers["robot/joint_states"].messages) == 5


def test_images_disabled_still_reports_camera_provenance_without_copying(ros):
    class Uncopyable:
        def copy(self, **kwargs):
            raise AssertionError("Disabled images must not be copied")

    pair = CameraPair(Uncopyable(), Uncopyable(), 100, 200, 31.5, 41.5)
    with Ros2Publisher(config(publish_images=False)) as bridge:
        bridge.submit(state(), cameras=pair)
        metadata = json.loads(ros.wait_for("timestamps").data)
        assert not any(topic.startswith("camera/") for topic in ros.publishers)
        assert metadata["cameras"]["external"]["sensor_timestamp"] == 31.5
        assert metadata["cameras"]["wrist"]["host_capture_monotonic_ns"] == 200
        assert not metadata["cameras"]["external"]["image_published"]


def test_mutable_inputs_copied_and_image_encoding_only_on_worker(ros, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    encode_threads = []
    original_publish = Ros2Publisher._publish

    def blocked_publish(bridge, snapshot):
        entered.set()
        assert release.wait(3)
        original_publish(bridge, snapshot)

    class TrackedImage(np.ndarray):
        def tobytes(self, *args, **kwargs):
            encode_threads.append(threading.get_ident())
            return super().tobytes(*args, **kwargs)

    monkeypatch.setattr(Ros2Publisher, "_publish", blocked_publish)
    robot, raw, pair = state(), vr(), cameras()
    pair = CameraPair(
        pair.front.view(TrackedImage),
        pair.wrist.view(TrackedImage),
        pair.front_ns,
        pair.wrist_ns,
        pair.front_sensor_timestamp,
        pair.wrist_sensor_timestamp,
    )
    original_rgb = np.asarray(pair.front).tobytes()
    target = (np.array([0.6, 0.7, 0.8]), np.array([0.0, 0.0, 0.0, 1.0]))
    with Ros2Publisher(config()) as bridge:
        try:
            bridge.submit(robot, vr=raw, target=target, cameras=pair)
            assert entered.wait(3)
            assert encode_threads == []
            robot.joint_positions[:] = 99
            robot.ee_position[:] = 99
            robot.gripper.commanded_open_ratio = 0
            raw.pos_x, raw.pose_seq = 99, 99
            raw.raw_position[:] = [99, 99, 99]
            pair.front[:] = 0
            pair.wrist[:] = 0
            target[0][:] = 99
            target[1][:] = 99
        finally:
            release.set()
        ros.wait_for("timestamps")
        assert ros.wait_for("robot/joint_states").position == list(range(7))
        assert ros.wait_for("robot/ee_pose").pose.position.x == 0.4
        assert json.loads(ros.wait_for("robot/gripper_state").data)["commanded_open_ratio"] == 0.7
        copied_vr = json.loads(ros.wait_for("vr/input").data)
        assert copied_vr["position"][0] == 0.1 and copied_vr["pose_seq"] == 42
        assert copied_vr["raw_position"] == [0.12, 0.21, 0.31]
        assert ros.wait_for("robot/target_ee_pose").pose.position.x == 0.6
        assert ros.wait_for("robot/target_ee_pose").pose.orientation.w == 1.0
        assert ros.wait_for("camera/external/image").data == original_rgb
        assert encode_threads == [bridge._thread.ident, bridge._thread.ident]


def test_worker_keeps_only_latest_pending_snapshot_when_dds_blocks(ros):
    entered, release = threading.Event(), threading.Event()

    def block(_message):
        entered.set()
        assert release.wait(3)

    ros.hooks["robot/joint_states"] = block
    with Ros2Publisher(config()) as bridge:
        try:
            bridge.submit(state(100))
            assert entered.wait(3)
            for ns in range(101, 301):
                bridge.submit(state(ns))
            assert bridge._pending.robot["sampled_monotonic_ns"] == 300
            assert ros.publishers["timestamps"].messages == []
        finally:
            release.set()
        first = json.loads(ros.wait_for("timestamps").data)
        last = json.loads(ros.wait_for("timestamps", 2).data)
        assert first["robot"]["sampled_monotonic_ns"] == 100
        assert last["robot"]["sampled_monotonic_ns"] == 300
        assert len(ros.publishers["robot/joint_states"].messages) == 2


def test_worker_rate_limits_new_snapshots_and_does_not_repeat_without_submit(ros):
    with Ros2Publisher(config(publish_hz=10)) as bridge:
        bridge.submit(state(100))
        ros.wait_for("timestamps")
        for ns in range(101, 201):
            bridge.submit(state(ns))
        last = json.loads(ros.wait_for("timestamps", 2).data)
        times = ros.publishers["timestamps"].publish_times
        assert times[1] - times[0] >= 0.08
        assert last["robot"]["sampled_monotonic_ns"] == 200
        with ros.condition:
            assert not ros.condition.wait_for(
                lambda: len(ros.publishers["timestamps"].messages) > 2, timeout=0.12
            )


def test_vr_startup_timestamp_preserved_with_explicit_header_fallback(ros, monkeypatch):
    monkeypatch.setattr(ros2_bridge.time, "monotonic_ns", lambda: 9_000_000_000)
    with Ros2Publisher(config()) as bridge:
        bridge.submit(state(), vr=vr(0))
        metadata = json.loads(ros.wait_for("timestamps").data)
        assert metadata["vr"]["pose_monotonic_ns"] == 0
        assert metadata["vr"]["stamp_monotonic_ns"] == 9_000_000_000
        assert stamp_ns(ros.wait_for("vr/pose")) == ros.system_ns


@pytest.mark.parametrize(
    "missing", ["rclpy", "geometry_msgs.msg", "sensor_msgs.msg", "std_msgs.msg"]
)
def test_missing_dependencies_fail_at_construction_with_actionable_message(
    ros, monkeypatch, missing
):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == missing:
            raise ModuleNotFoundError(f"No module named {missing}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="ros2.enabled=false") as error:
        Ros2Publisher(config())
    assert "source /opt/ros" in str(error.value)
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert ros.contexts == []


@pytest.mark.parametrize("stage", ["init", "node", "publisher", "thread"])
def test_partial_initialization_always_cleans_up(ros, monkeypatch, stage):
    if stage == "init":
        ros.init_error = RuntimeError("init failed")
    elif stage == "node":
        ros.node_error = RuntimeError("node failed")
    elif stage == "publisher":
        ros.publisher_error_at = 3
    else:

        def fail_start(_thread):
            raise RuntimeError("thread start failed")

        monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="ROS 2 telemetry initialization failed"):
        Ros2Publisher(config())
    assert len(ros.contexts) == 1
    assert ros.contexts[0].shutdown_calls == ros.contexts[0].destroy_calls == 1
    assert not ros.contexts[0].initialized
    if ros.nodes:
        assert ros.nodes[0].destroy_calls == 1


def test_worker_failure_logs_disables_and_cleans_up_without_raising(ros, caplog):
    def fail(_message):
        raise RuntimeError("DDS publish failed")

    ros.hooks["robot/joint_states"] = fail
    bridge = Ros2Publisher(config())
    bridge.submit(state())
    bridge._thread.join(3)
    assert not bridge._thread.is_alive()
    assert bridge.enabled is False
    assert "telemetry worker failed" in caplog.text and "DDS publish failed" in caplog.text
    bridge.submit(object(), cameras=object())
    bridge.close()
    bridge.close()
    assert ros.nodes[0].destroy_calls == 1
    assert ros.contexts[0].shutdown_calls == ros.contexts[0].destroy_calls == 1


def test_bad_snapshot_disables_without_breaking_caller(ros, caplog):
    with Ros2Publisher(config()) as bridge:
        bridge.submit(object())
        bridge._thread.join(3)
        assert bridge.enabled is False
        assert "telemetry snapshot failed" in caplog.text
    assert ros.contexts[0].destroy_calls == 1


def test_cleanup_continues_if_node_and_context_shutdown_raise(ros, caplog):
    ros.destroy_error = ros.shutdown_error = True
    bridge = Ros2Publisher(config())
    bridge.close()
    bridge.close()
    assert ros.nodes[0].destroy_calls == 1
    assert ros.contexts[0].shutdown_calls == ros.contexts[0].destroy_calls == 1
    assert "node destroy failed" in caplog.text and "context shutdown failed" in caplog.text


def test_close_does_not_destroy_native_resources_while_worker_is_publishing(ros, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def block(_message):
        entered.set()
        assert release.wait(3)

    ros.hooks["robot/joint_states"] = block
    monkeypatch.setattr(Ros2Publisher, "_CLOSE_TIMEOUT_S", 0.01)
    bridge = Ros2Publisher(config())
    try:
        bridge.submit(state())
        assert entered.wait(3)
        bridge.submit(state(200))
        bridge.close()
        assert bridge.enabled is False and bridge._pending is None
        assert bridge._thread.is_alive()
        assert ros.nodes[0].destroy_calls == ros.contexts[0].shutdown_calls == 0
        bridge.submit(object())
    finally:
        release.set()
        bridge._thread.join(3)
        bridge.close()
    assert not bridge._thread.is_alive()
    assert len(ros.publishers["robot/joint_states"].messages) == 1
    assert ros.nodes[0].destroy_calls == 1
    assert ros.contexts[0].shutdown_calls == ros.contexts[0].destroy_calls == 1


def test_context_manager_preserves_application_exception(ros):
    with pytest.raises(ValueError, match="application failure"):
        with Ros2Publisher(config()):
            raise ValueError("application failure")
    assert ros.contexts[0].destroy_calls == 1


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_rate_fails_before_ros_initialization(ros, rate):
    with pytest.raises(ValueError, match="ros2.publish_hz"):
        Ros2Publisher(config(publish_hz=rate))
    assert ros.contexts == []
