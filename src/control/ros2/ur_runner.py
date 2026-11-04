"""Experimental six-axis UR runner over an independently started ROS2 driver."""

import threading
import time
from dataclasses import replace

import numpy as np

from control.base_runner import BaseRunner
from control.ros2.gripper_controller import Ros2GripperController
from control.ros2.subscription import Subscription
from control.ros2.ur_config import UR_JOINT_NAMES
from control.ros2.ur_kinematics import URKinematics
from control.types import RobotState
from control.util.pose import matrix_to_quat_xyzw


class URRos2Backend:
    finite_motion = False

    def __init__(self, config):
        self.config, self.cfg = config, config.get("ros2", {})
        self.namespace = self.cfg.get("namespace", "").strip("/")
        self.frame = self.cfg.get("base_frame", "base_link")
        self.names = list(self.cfg.get("joint_names", UR_JOINT_NAMES))
        self.timeout = float(self.cfg.get("state_timeout_s", 0.25))
        self.duration = float(self.cfg.get("command_duration_s", 0.15))
        self._source = Subscription("data_collect_ur", self.topic(""), monitoring=self.cfg.get("monitoring"))
        self._state = self._model = self.gripper = None
        self._description = None
        self._source_ns = 0
        self._streaming = False
        self._cancel = threading.Event()

    def topic(self, name):
        return "/" + "/".join(part for part in (self.namespace, name.strip("/")) if part)

    @property
    def _node(self):
        return self._source.node

    def connect(self):
        if self._node is not None:
            self.snapshot()
            return
        try:
            self._source.start(self._setup)
            self._source.wait(lambda: self._state is not None, float(self.cfg.get("connect_timeout_s", 10)))
            self.snapshot()
        except BaseException:
            self.close()
            raise

    def _setup(self, node):
        from controller_manager_msgs.srv import ListControllers
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        self._trajectory_type, self._point_type = JointTrajectory, JointTrajectoryPoint
        self._list_type = ListControllers
        self._controllers = node.create_client(ListControllers, self.topic("controller_manager/list_controllers"))
        self._publisher = node.create_publisher(
            JointTrajectory, self.topic(self.cfg.get("trajectory_topic", "scaled_joint_trajectory_controller/joint_trajectory")),
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.gripper = Ros2GripperController(self.config, node, source=self._source)
        try:
            self.gripper.connect()
        except BaseException:
            self.gripper.close()
            self.gripper = None
            raise
        self._source.subscribe(
            String, self.topic(self.cfg.get("robot_description_topic", "robot_description")), self._on_description,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE),
        )
        self._source.subscribe(
            JointState, self.topic(self.cfg.get("joint_state_topic", "joint_states")), self._on_state,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            expected_hz=float(self.config.get("control", {}).get("frequency_hz", 100)), timeout_s=self.timeout,
        )

    def _on_description(self, msg):
        try:
            if self._description is not None:
                if msg.data != self._description:
                    raise ValueError("UR robot_description changed; reconnect the runner")
                return
            model = URKinematics(msg.data, self.frame, self.cfg.get("tool_frame", "tool0"), self.names)
            with self._source.condition:
                self._model, self._description = model, msg.data
        except (ValueError, TypeError, AttributeError) as exc:
            self._source.failure = exc

    def _on_state(self, msg):
        if self._model is None:
            return False
        try:
            source_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            age = (self._node.get_clock().now().nanoseconds - source_ns) / 1e9
            if source_ns <= self._source_ns or not -0.1 <= age <= self.timeout:
                raise ValueError("UR joint feedback is stale or clocks are unsynchronized")
            if len(set(msg.name)) != len(msg.name):
                raise ValueError("Duplicate UR joint names")
            order = [msg.name.index(name) for name in self.names]
            joints = np.array([msg.position[index] for index in order])
            velocities = np.array([msg.velocity[index] for index in order])
            pose = self._model.forward(joints)
            state = RobotState(joints, velocities, pose[:3, 3], matrix_to_quat_xyzw(pose[:3, :3]),
                               pose, self.gripper.get_state(), time.monotonic_ns(),
                               base_frame=self.frame, robot_time_s=source_ns / 1e9)
        except (ValueError, TypeError, IndexError) as exc:
            self._node.get_logger().warning(str(exc), throttle_duration_sec=2.0)
            return False
        with self._source.condition:
            self._state, self._source_ns = state, source_ns
            self._source.condition.notify_all()

    def snapshot(self, busy=False):
        with self._source.condition:
            self._source.check()
            if self._state is None:
                raise RuntimeError("No UR state/robot_description received")
            if time.monotonic_ns() - self._state.sampled_monotonic_ns > self.timeout * 1e9:
                raise TimeoutError("UR feedback is stale; refusing commands")
            return replace(self._state, gripper=self.gripper.get_state(busy=busy))

    def _require_controller(self):
        timeout = float(self.cfg.get("connect_timeout_s", 10))
        deadline = time.monotonic() + timeout
        if not self._controllers.wait_for_service(timeout_sec=timeout):
            raise TimeoutError("UR controller manager unavailable")
        name = self.cfg.get("controller_name", "scaled_joint_trajectory_controller")
        while time.monotonic() < deadline:
            self._source.check()
            future = self._controllers.call_async(self._list_type.Request())
            done = threading.Event()
            future.add_done_callback(lambda _: done.set())
            if not done.wait(max(0, deadline - time.monotonic())):
                future.cancel()
                raise TimeoutError("UR controller status request timed out")
            if any(item.name == name and item.state == "active" for item in future.result().controller):
                return
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        raise RuntimeError(f"UR controller {name} is not active; start deploy_ros2.sh --ur first")

    def _publish_joints(self, target, duration=None):
        self.snapshot()
        target = self._model.validate(target)
        if self._publisher.get_subscription_count() == 0:
            raise RuntimeError("No UR joint trajectory subscriber")
        msg = self._trajectory_type()
        msg.joint_names = self.names
        point = self._point_type()
        point.positions = list(map(float, target))
        # Zero timestamp starts immediately; the controller holds the finite endpoint.
        ns = round((self.duration if duration is None else duration) * 1e9)
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(ns, 1_000_000_000)
        msg.points = [point]
        self._publisher.publish(msg)

    def start(self, state):
        self._require_controller()
        self._publish_joints(self.snapshot().joint_positions)
        self._streaming = True

    def target(self, position, quaternion, nullspace):
        if not self._streaming:
            raise RuntimeError("UR EE stream is not started")
        state = self.snapshot()
        target = self._model.inverse(position, quaternion, state.joint_positions)
        delta = target - state.joint_positions
        limit = float(self.cfg.get("max_command_joint_error_rad", 0.15))
        if np.max(np.abs(delta)) > limit:
            raise ValueError("UR IK target exceeds measured joint guard")
        speed = np.minimum(self._model.velocity, float(self.cfg.get("max_joint_velocity_rad_s", 0.3)))
        scale = max(1.0, float(np.max(np.abs(delta) / (speed * self.duration))))
        self._publish_joints(state.joint_positions + delta / scale)

    def stop(self):
        if self._streaming:
            self._publish_joints(self.snapshot().joint_positions)
        self._streaming = False

    def move_joints(self, target, speed):
        self._require_controller()
        target = self._model.validate(target)
        self._cancel.clear()
        initial = self.snapshot().joint_positions
        velocity = min(float(self.cfg.get("max_joint_velocity_rad_s", 0.3)), float(np.min(self._model.velocity))) * speed
        duration = max(self.duration, float(np.max(np.abs(target - initial))) / velocity)
        started = time.monotonic()
        try:
            while not self._cancel.is_set():
                elapsed = time.monotonic() - started
                fraction = min(1.0, elapsed / duration)
                state = self.snapshot()
                next_target = initial + fraction * (target - initial)
                if np.max(np.abs(next_target - state.joint_positions)) > float(self.cfg.get("max_command_joint_error_rad", 0.15)):
                    raise RuntimeError("UR joint motion tracking error exceeds guard")
                self._publish_joints(next_target)
                if fraction == 1 and np.max(np.abs(state.joint_positions - target)) < float(self.config.get("robot", {}).get("joint_tolerance_rad", 0.02)):
                    return True
                if elapsed > duration + float(self.config.get("robot", {}).get("arrival_timeout_s", 5)):
                    raise TimeoutError("UR joints did not reach requested target")
                self._cancel.wait(0.02)
            return False
        finally:
            self._publish_joints(self.snapshot().joint_positions)

    def move_ee(self, position, quaternion, speed):
        return self.move_joints(self._model.inverse(position, quaternion, self.snapshot().joint_positions), speed)

    def pose_from_joints(self, joints):
        return self._model.forward(joints)

    def cancel(self):
        self._cancel.set()
        self._publish_joints(self.snapshot().joint_positions)

    def gripper_command(self, ratio, *, speed, force):
        return self.gripper.command(ratio, speed=speed, force=force)

    def stop_gripper(self):
        return self.gripper.stop()

    def tactile_frames(self):
        return self.gripper.tactile_frames()

    def tactile_images(self):
        return self.gripper.tactile_images()

    def close(self):
        try:
            self.stop()
        finally:
            self._source.close(before_destroy=self.gripper.close if self.gripper is not None else None)
            self.gripper = self._state = self._model = self._description = None
            self._source_ns = 0
            self._streaming = False


class URRunner(BaseRunner):
    def _make_backend(self, config):
        return URRos2Backend(config)
