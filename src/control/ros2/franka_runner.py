"""Franka ROS 2 backend. No SDK connections or realtime control in Python."""

from __future__ import annotations

import time

import numpy as np

from control.base_runner import BaseRunner
from control.ros2.gripper_controller import Ros2GripperController
from control.ros2.subscription import Subscription
from control.types import RobotState
from control.util.pose import matrix_to_quat_xyzw, normalize_quat_xyzw, position_quat_to_matrix

_ROBOT_MODES = (
    "Other",
    "Idle",
    "Move",
    "Guiding",
    "Reflex",
    "UserStopped",
    "AutomaticErrorRecovery",
)


def _robot_mode_name(mode):
    if isinstance(mode, str):
        return mode
    try:
        return _ROBOT_MODES[int(mode)]
    except (TypeError, ValueError, IndexError):
        return str(mode)


class Ros2Backend:
    finite_motion = False

    def __init__(self, config):
        self.config = config
        self.cfg = config.get("ros2", {})
        self._ee_wrench_enabled = config.get("observation", {}).get("ee_wrench", False)
        self.namespace = self.cfg.get("namespace", "").strip("/")
        self.frame = self.cfg.get("base_frame", "panda_link0")
        self.timeout = float(self.cfg.get("state_timeout_s", 0.25))
        self._source = Subscription(
            "data_collect_control", self.topic(""), monitoring=self.cfg.get("monitoring")
        )
        self._condition = self._source.condition
        self._state = None
        self._streaming = False
        self.gripper = None

    def topic(self, name):
        return "/" + "/".join(p for p in (self.namespace, name.strip("/")) if p)

    def connect(self):
        if self._source.node is not None:
            self._source.check()
            return
        try:
            self._source.start(self._setup)
            self._source.wait(
                lambda: self._state is not None, float(self.cfg.get("connect_timeout_s", 10))
            )
            self.snapshot()
        except BaseException:
            self.close()
            raise

    @property
    def _node(self):
        return self._source.node

    def _setup(self, node):
        from franka_msgs.msg import FrankaRobotState
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        self._pose_type = PoseStamped
        self._publisher = node.create_publisher(
            PoseStamped,
            self.topic(self.cfg.get(
                "equilibrium_pose_topic", "cartesian_pose_impedance_controller/equilibrium_pose"
            )),
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._subscription = self._source.subscribe(
            FrankaRobotState,
            self.topic(self.cfg.get("robot_state_topic", "franka_robot_state_broadcaster/robot_state")),
            self._on_state,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            expected_hz=float(self.config.get("control", {}).get("frequency_hz", 100)),
            timeout_s=self.timeout,
        )
        self.gripper = Ros2GripperController(self.config, node, source=self._source)
        try:
            self.gripper.connect()
        except BaseException:
            # Subscription.start destroys its node on setup failure. Release
            # partially connected clients while their node still exists.
            self.gripper.close()
            self.gripper = None
            raise

    def _on_state(self, msg):
        try:
            # FER libfranka 0.9 publishes O_T_EE as a column-major 4x4 in the base frame.
            frame = msg.header.frame_id
            if frame and frame != self.frame:
                raise ValueError(f"Expected EE frame {self.frame}, got {frame!r}")
            stamp = msg.header.stamp
            source_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            age = (self._node.get_clock().now().nanoseconds - source_ns) / 1e9
            if source_ns <= 0 or not -0.1 <= age <= self.timeout:
                raise ValueError("Franka source timestamp is stale or clocks are unsynchronized")
            matrix = np.asarray(msg.o_t_ee, dtype=np.float64).reshape((4, 4), order="F")
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError("Franka O_T_EE must be a finite column-major 4x4")
            position = matrix[:3, 3].copy()
            quaternion = matrix_to_quat_xyzw(matrix[:3, :3])
            joints = np.asarray(msg.q, dtype=np.float64)
            velocities = np.asarray(msg.dq, dtype=np.float64)
            if (
                joints.shape != (7,)
                or velocities.shape != (7,)
                or not np.isfinite(joints).all()
                or not np.isfinite(velocities).all()
            ):
                raise ValueError("Franka joint state must contain 7 finite positions and velocities")
            state = RobotState(
                joints,
                velocities,
                position,
                quaternion,
                position_quat_to_matrix(position, quaternion),
                self.gripper.get_state(),
                time.monotonic_ns(),
                base_frame=self.frame,
                robot_time_s=float(msg.time),
                robot_mode=_robot_mode_name(msg.robot_mode),
                control_command_success_rate=float(msg.control_command_success_rate),
                current_errors=str(msg.current_errors),
                ee_wrench=(
                    np.asarray(getattr(msg, "o_f_ext_hat_k", None), dtype=np.float64)
                    if self._ee_wrench_enabled else None
                ),
            )
        except (ValueError, IndexError, TypeError) as exc:
            # Invalid messages do not refresh the last valid feedback receipt.
            self._node.get_logger().warning(str(exc), throttle_duration_sec=2.0)
            return False
        with self._condition:
            self._state = state
            self._condition.notify_all()

    def snapshot(self, busy=False):
        from dataclasses import replace

        with self._condition:
            self._source.check()
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
        self._source.close(before_destroy=self.gripper.close if self.gripper is not None else None)
        self.gripper = None
        self._state = None


class FrankaRunner(BaseRunner):
    """ROS state subscription, EE publication and gripper actions.

    The official realtime controller is started separately by bringup.sh.
    """

    def _make_backend(self, config: dict):
        return Ros2Backend(config)

