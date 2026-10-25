"""Franka ROS 2 backend. No SDK connections or realtime control in Python."""

from __future__ import annotations

import threading
import time

import numpy as np

from control.base_runner import BaseRunner
from control.ros2.gripper_controller import Ros2GripperController
from control.types import RobotState
from control.util.pose import normalize_quat_xyzw, position_quat_to_matrix


class Ros2Backend:
    finite_motion = False

    def __init__(self, config):
        self.config = config
        self.cfg = config.get("ros2", {})
        self.namespace = self.cfg.get("namespace", "").strip("/")
        self.frame = self.cfg.get("base_frame", "fr3_link0")
        self.timeout = float(self.cfg.get("state_timeout_s", 0.25))
        self._condition = threading.Condition()
        self._context = self._node = self._executor = self._thread = None
        self._state = self._failure = None
        self._streaming = False
        self.gripper = None

    def topic(self, name):
        return "/" + "/".join(p for p in (self.namespace, name.strip("/")) if p)

    def connect(self):
        import rclpy
        from franka_msgs.msg import FrankaRobotState
        from geometry_msgs.msg import PoseStamped
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions

        if self._node is not None:
            return
        try:
            self._context = Context()
            rclpy.init(
                args=[], context=self._context, signal_handler_options=SignalHandlerOptions.NO
            )
            self._node = rclpy.create_node(
                "data_collect_control",
                namespace=self.topic(""),
                context=self._context,
                use_global_arguments=False,
            )
            self._pose_type = PoseStamped
            self._publisher = self._node.create_publisher(
                PoseStamped,
                self.topic(
                    self.cfg.get(
                        "equilibrium_pose_topic",
                        "cartesian_impedance_example_controller/equilibrium_pose",
                    )
                ),
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            )
            self._subscription = self._node.create_subscription(
                FrankaRobotState,
                self.topic(
                    self.cfg.get("robot_state_topic", "franka_robot_state_broadcaster/robot_state")
                ),
                self._on_state,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            )
            self.gripper = Ros2GripperController(self.config, self._node)
            self.gripper.connect()
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            self._thread = threading.Thread(target=self._spin, name="ros2-feedback", daemon=True)
            self._thread.start()
            deadline = time.monotonic() + float(self.cfg.get("connect_timeout_s", 10))
            with self._condition:
                while self._state is None and self._failure is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "No valid Franka ROS state; check namespace and broadcaster"
                        )
                    self._condition.wait(remaining)
            self.snapshot()
        except BaseException:
            self.close()
            raise

    def _spin(self):
        try:
            self._executor.spin()
        except Exception as exc:
            with self._condition:
                self._failure = exc
                self._condition.notify_all()

    def _on_state(self, msg):
        try:
            pose = msg.o_t_ee
            if pose.header.frame_id != self.frame:
                raise ValueError(f"Expected EE frame {self.frame}, got {pose.header.frame_id!r}")
            stamp = pose.header.stamp
            source_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            age = (self._node.get_clock().now().nanoseconds - source_ns) / 1e9
            if source_ns <= 0 or not -0.1 <= age <= self.timeout:
                raise ValueError("Franka source timestamp is stale or clocks are unsynchronized")
            p, q = pose.pose.position, pose.pose.orientation
            position = np.array([p.x, p.y, p.z])
            quaternion = normalize_quat_xyzw([q.x, q.y, q.z, q.w])
            joints = msg.measured_joint_state
            names = [f"{self.cfg.get('arm_id', 'fr3')}_joint{i}" for i in range(1, 8)]
            indices = [list(joints.name).index(name) for name in names]
            state = RobotState(
                np.array(joints.position)[indices],
                np.array(joints.velocity)[indices],
                position,
                quaternion,
                position_quat_to_matrix(position, quaternion),
                self.gripper.get_state(),
                time.monotonic_ns(),
                base_frame=self.frame,
                robot_time_s=float(msg.time),
                robot_mode=str(msg.robot_mode),
                control_command_success_rate=float(msg.control_command_success_rate),
                current_errors=str(msg.current_errors),
            )
        except (ValueError, IndexError, TypeError) as exc:
            # Invalid messages do not refresh the last valid feedback receipt.
            self._node.get_logger().warning(str(exc), throttle_duration_sec=2.0)
            return
        with self._condition:
            self._state = state
            self._condition.notify_all()

    def snapshot(self, busy=False):
        from dataclasses import replace

        with self._condition:
            if self._failure is not None:
                raise RuntimeError("ROS feedback executor failed") from self._failure
            if self._state is None:
                raise RuntimeError("No Franka state received")
            if time.monotonic_ns() - self._state.sampled_monotonic_ns > self.timeout * 1e9:
                raise TimeoutError("Franka feedback is stale; refusing commands")
            return replace(self._state, gripper=self.gripper.get_state(busy=busy))

    def start(self, state):
        self._streaming = True
        try:
            self.target(state.ee_position, state.ee_quaternion_xyzw, None)
        except BaseException:
            self._streaming = False
            raise

    def target(self, position, quaternion, nullspace):
        if not self._streaming:
            raise RuntimeError("ROS EE stream is not started")
        self.snapshot()
        if self._publisher.get_subscription_count() == 0:
            raise RuntimeError("No equilibrium_pose subscriber; activate the official controller")
        msg = self._pose_type()
        msg.header.frame_id = self.frame
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, position)
        q = normalize_quat_xyzw(quaternion)
        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = map(float, q)
        self._publisher.publish(msg)

    def stop(self):
        # The owner publishes a measured hold before this method. The C++
        # impedance controller stays active and retains its equilibrium target.
        self._streaming = False

    def move_joints(self, target, speed):
        raise NotImplementedError(
            "Equilibrium PoseStamped cannot command joints; initialize in Franka bringup"
        )

    def move_ee(self, position, quaternion, speed):
        raise NotImplementedError("Use execute_ee_trajectory for bounded Cartesian motion")

    def pose_from_joints(self, joints):
        raise NotImplementedError("ROS backend does not implement FK; use robot_state_publisher/TF")

    def cancel(self):
        state = self.snapshot()
        if self._streaming:
            self.target(state.ee_position, state.ee_quaternion_xyzw, None)

    def gripper_command(self, ratio, *, speed, force):
        return self.gripper.command(ratio, speed=speed, force=force)

    def stop_gripper(self):
        return self.gripper.stop()

    def tactile_frames(self):
        return self.gripper.tactile_frames()

    def tactile_images(self):
        return self.gripper.tactile_images()

    def close(self):
        self._streaming = False
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=2)
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("ROS executor did not stop; resources retained")
        if self.gripper is not None:
            self.gripper.close()
        if self._node is not None:
            self._node.destroy_node()
        if self._context is not None:
            self._context.try_shutdown()
            self._context.destroy()
        self._context = self._node = self._executor = self._thread = None
        self._state = self._failure = None


class FrankaRunner(BaseRunner):
    """ROS state subscription, EE publication and gripper actions.

    The official realtime controller is started separately by bringup.sh.
    """

    def _make_backend(self, config: dict):
        return Ros2Backend(config)

