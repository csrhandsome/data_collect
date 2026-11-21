"""Drive a standalone DH5 gripper from the VR control client."""

import json
import math
import os
import time
from types import SimpleNamespace

import rclpy
from franka_msgs.action import Move
from rclpy._rclpy_pybind11 import RCLError
from rclpy.action import ActionServer
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String, UInt16

from control.config import DEFAULT_CONFIG, load_config
from control.direct.grippers.dh5 import DH5GripperBackend
from control.util.robot import DEFAULT_GRIPPER_AXIS_DEADZONE

COMMAND_TOPIC = "/dh5/vr_command"
RUNNER_COMMAND_TOPIC = "/dh5/runner_vr_command"


def main(argv=None):
    config = load_config(os.environ.get("DATA_COLLECT_CONFIG", DEFAULT_CONFIG))
    gripper_config = config.get("gripper", {})
    options = dict(gripper_config.get("dh5", {}))
    for key in ("discrete_level_count", "vr_axis_threshold", "vr_step_interval_s"):
        if key in gripper_config:
            options[key] = gripper_config[key]
    deadzone = float(options.get("vr_axis_threshold", DEFAULT_GRIPPER_AXIS_DEADZONE))
    open_width = float(gripper_config.get("open_width_m", 0.05))

    rclpy.init(args=argv)
    node = None
    backend = action = None
    try:
        node = Node("dh5_gripper")
        options["com"] = node.declare_parameter("com", options.get("com", "/dev/ttyUSB0")).value
        targets = node.create_publisher(UInt16, "/dh5/target_position", 1)
        states = node.create_publisher(String, "/dh5/state", 1)
        backend = DH5GripperBackend({
            **config, "gripper": {**gripper_config, "dh5": options, "open_position_high": True},
        })
        backend.connect()
        gripper = backend.device
        if gripper is not None:
            velocity = int(gripper_config.get("velocity", 100))
            position_synced = False
            command_error = None
            last_seq = last_axis = None
            last_absolute_ns = 0
            absolute_active = False

            def record_command_error(exc):
                nonlocal absolute_active, command_error, position_synced
                command_error = str(exc)
                absolute_active = False
                position_synced = False
                node.get_logger().warning(f"DH5 position command: {exc}")

            def on_vr(message):
                nonlocal command_error, last_seq, last_axis
                try:
                    payload = json.loads(message.data)
                    sample = SimpleNamespace(
                        pose_monotonic_ns=int(payload["pose_monotonic_ns"]),
                        pose_seq=int(payload["pose_seq"]),
                        gripper_velocity_axis=float(payload["gripper_velocity_axis"]),
                    )
                    if not math.isfinite(sample.gripper_velocity_axis):
                        raise ValueError("Nonfinite gripper axis")
                except (KeyError, TypeError, ValueError, OverflowError):
                    node.get_logger().warning("Invalid VR sample", throttle_duration_sec=2.0)
                    return
                last_seq, last_axis = sample.pose_seq, sample.gripper_velocity_axis
                if absolute_active or sample.pose_monotonic_ns <= last_absolute_ns:
                    return
                if not position_synced:
                    return
                if command_error is not None:
                    age_ns = time.monotonic_ns() - sample.pose_monotonic_ns
                    if abs(last_axis) <= deadzone and 0 <= age_ns < 500_000_000:
                        command_error = None
                        node.get_logger().info("DH5 feedback recovered and stick centered; control enabled")
                    return
                # The standalone DH5's native register increases toward opening.
                sample.gripper_velocity_axis = -last_axis
                try:
                    if gripper.update_from_vr(sample, wait=True):
                        targets.publish(UInt16(data=gripper.position_command))
                except (OSError, RuntimeError) as exc:
                    record_command_error(exc)

            def move(goal):
                nonlocal absolute_active, last_absolute_ns
                result = Move.Result()
                width = float(goal.request.width)
                if not position_synced or not 0 <= width <= open_width:
                    result.error = "DH5 unavailable or requested width outside configured range"
                    goal.abort()
                    return result
                last_absolute_ns = time.monotonic_ns()
                try:
                    backend.command(width / open_width, speed=goal.request.speed, force=0.0)
                except (OSError, RuntimeError) as exc:
                    record_command_error(exc)
                    result.error = str(exc)
                    goal.abort()
                    return result
                targets.publish(UInt16(data=gripper.position_command))
                absolute_active = True
                result.success = True
                goal.succeed()
                return result

            def publish_state():
                nonlocal absolute_active, position_synced
                measurement = {
                    "sample_ns": time.monotonic_ns(),
                    "received_seq": last_seq, "received_axis": last_axis,
                    "applied_axis": -last_axis if last_axis is not None else None,
                    "actual_position": None, "grip_status": None,
                    "velocity_setting": velocity, "feedback_error": None,
                    "command_error": command_error,
                }
                try:
                    if not position_synced:
                        gripper.Hand.clear_input_buffer()
                    position = gripper.actual_position
                    measurement["actual_position"] = position
                    if not position_synced:
                        gripper.sync_position_command(position)
                        position_synced = True
                    measurement["grip_status"] = gripper.grip_status
                    if absolute_active and (
                        position == gripper.position_command or measurement["grip_status"] in (1, 2, 3)
                    ):
                        absolute_active = False
                except (OSError, RuntimeError) as exc:
                    measurement["feedback_error"] = str(exc)
                    node.get_logger().warning(f"DH5 feedback: {exc}", throttle_duration_sec=2.0)
                measurement["target_position"] = gripper.position_command if position_synced else None
                measurement["control_state"] = (
                    "waiting_for_feedback" if not position_synced
                    else "waiting_for_neutral" if command_error is not None
                    else "ready"
                )
                measurement["absolute_active"] = absolute_active
                states.publish(String(data=json.dumps(measurement)))

            runner = node.create_subscription(String, RUNNER_COMMAND_TOPIC, on_vr, 1)

            def on_manual_vr(message):
                if runner.get_publisher_count() == 0:
                    on_vr(message)

            node.create_subscription(String, COMMAND_TOPIC, on_manual_vr, 1)
            action = ActionServer(node, Move, "/dh5/move", move)
            node.create_timer(0.1, publish_state)
            if backend.enable_cameras:
                start_camera_publishers(node, backend, options)
            node.get_logger().info(f"DH5 on {options['com']}, listening to {COMMAND_TOPIC}")
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():
            raise
    finally:
        if action is not None:
            action.destroy()
        if backend is not None:
            backend.close()
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


def start_camera_publishers(node, backend, options):
    sides = ("left", "right")
    publishers = [
        node.create_publisher(Image, f"/dh5/{side}/image_raw", qos_profile_sensor_data)
        for side in sides
    ]
    sent = [None, None]

    def publish():
        left, right, left_ns, right_ns = backend.tactile_frames()
        for index, (frame, capture_ns) in enumerate(((left, left_ns), (right, right_ns))):
            if frame is None or capture_ns is None or capture_ns == sent[index]:
                continue
            if publishers[index].get_subscription_count() == 0:
                continue
            msg = Image()
            clock_ns = node.get_clock().now().nanoseconds
            stamp_ns = clock_ns - (time.monotonic_ns() - capture_ns)
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
            msg.header.frame_id = f"dh5_{sides[index]}_camera_optical_frame"
            msg.height, msg.width = frame.shape[:2]
            msg.encoding = "rgb8"
            msg.step = msg.width * 3
            msg.data = frame.tobytes()
            publishers[index].publish(msg)
            sent[index] = capture_ns

    node.create_timer(1 / max(float(options.get("camera_fps", 30)), 1), publish)


if __name__ == "__main__":
    main()
