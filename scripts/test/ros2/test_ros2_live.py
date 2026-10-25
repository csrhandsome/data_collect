"""Opt-in DDS tests: source ROS, then DATA_COLLECT_TEST_ROS2=1 python -m pytest this_file."""

import json
import os
import time
from uuid import uuid4

import numpy as np
import pytest

from control.collection.devices import FakeCameras, FakeVR
from control.config import load_config
from control.ros2_bridge import Ros2Publisher

pytestmark = pytest.mark.skipif(
    os.environ.get("DATA_COLLECT_TEST_ROS2") != "1",
    reason="Opt-in real ROS 2 communication test; requires a sourced ROS installation and DDS sockets",
)


@pytest.fixture
def subscriber():
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import String

    context = Context()
    rclpy.init(args=[], context=context)
    namespace = f"/test_data_collect_{uuid4().hex[:8]}"
    node = rclpy.create_node("observer", namespace=namespace, context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    received = {}
    topics = {
        "robot/joint_states": JointState,
        "robot/ee_pose": PoseStamped,
        "robot/gripper_state": String,
        "robot/target_ee_pose": PoseStamped,
        "vr/pose": PoseStamped,
        "vr/input": String,
        "camera/external/image": Image,
        "camera/wrist/image": Image,
        "timestamps": String,
    }
    subscriptions = [
        node.create_subscription(
            message_type,
            topic,
            lambda message, key=topic: received.setdefault(key, []).append(message),
            qos_profile_sensor_data,
        )
        for topic, message_type in topics.items()
    ]
    try:
        yield namespace, received, executor
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        context.shutdown()


def spin_until(executor, condition, *, timeout=8):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
    assert condition(), "ROS subscriber did not receive the expected telemetry before timeout"


def test_real_dds_payloads_and_timestamps(subscriber):
    from control._panda.fake import FakeBackend
    from control.robotic_arm_controller import RoboticArmControler

    namespace, received, executor = subscriber
    config = load_config("config/train/panda_ros2.yaml")
    config["ros2"]["telemetry_namespace"] = namespace
    with RoboticArmControler(config=config, backend=FakeBackend()) as arm:
        state = arm.get_state()
    pair = FakeCameras(config).get_frames()
    vr = FakeVR().latest
    vr.y_pressed = True
    with Ros2Publisher(config) as bridge:
        spin_until(
            executor,
            lambda: all(
                publisher.get_subscription_count() >= 1 for publisher in bridge._publishers.values()
            ),
        )
        deadline = time.monotonic() + 8
        while len(received) < 9 and time.monotonic() < deadline:
            bridge.submit(
                state, vr=vr, target=(state.ee_position, state.ee_quaternion_xyzw), cameras=pair
            )
            executor.spin_once(timeout_sec=0.02)
        assert len(received) == 9, f"Only received: {sorted(received)}"
        joints = received["robot/joint_states"][0]
        assert len(joints.name) == 7
        np.testing.assert_allclose(joints.position, state.joint_positions)
        np.testing.assert_allclose(joints.velocity, state.joint_velocities)
        pose = received["robot/ee_pose"][0]
        assert pose.header.frame_id == state.base_frame
        assert pose.pose.position.x == pytest.approx(state.ee_position[0])
        assert received["vr/pose"][0].header.frame_id == "vr_world"
        for key, expected in [("external", pair.front), ("wrist", pair.wrist)]:
            message = received[f"camera/{key}/image"][0]
            assert message.encoding == "rgb8"
            assert message.header.frame_id == f"{key}_camera_optical_frame"
            assert message.height == 224 and message.width == 224 and message.step == 224 * 3
            assert bytes(message.data) == expected.tobytes()
            # Headers must be epoch/system time, not time.monotonic_ns().
            assert abs(message.header.stamp.sec - time.time()) < 10
        gripper = json.loads(received["robot/gripper_state"][0].data)
        assert gripper["commanded_open_ratio"] == state.gripper.commanded_open_ratio
        assert gripper["sampled_monotonic_ns"] == state.sampled_monotonic_ns
        assert gripper["stamp"] == {
            "sec": pose.header.stamp.sec,
            "nanosec": pose.header.stamp.nanosec,
        }
        vr_payload = json.loads(received["vr/input"][0].data)
        assert vr_payload["pose_monotonic_ns"] == vr.pose_monotonic_ns
        assert vr_payload["pose_seq"] == vr.pose_seq
        assert vr_payload["buttons"]["y_pressed"] is True
        timestamps = json.loads(received["timestamps"][0].data)
        assert timestamps["robot"]["sampled_monotonic_ns"] == state.sampled_monotonic_ns
        assert timestamps["cameras"]["external"]["host_capture_monotonic_ns"] == pair.front_ns
        assert timestamps["cameras"]["wrist"]["sensor_timestamp"] == pair.wrist_sensor_timestamp
        assert timestamps["clock"]["header_clock"] == "ros_system_time"
        # Republishing the same camera sample must not emit duplicate images.
        end = time.monotonic() + 0.2
        while time.monotonic() < end:
            bridge.submit(state, vr=vr, cameras=pair)
            executor.spin_once(timeout_sec=0.01)
        assert len(received["camera/external/image"]) == 1
        assert len(received["camera/wrist/image"]) == 1
