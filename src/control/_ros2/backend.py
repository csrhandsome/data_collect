"""Franka ROS 2 backend. No SDK connections or realtime control in Python."""

from __future__ import annotations

import threading
import time

import numpy as np

from control.robotic_arm_types import GripperState, RobotState
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
        self._ratio = 1.0
        self._width = None
        self._clients = []

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
            self._setup_gripper()
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
                GripperState(
                    self.config.get("gripper", {}).get("type", "franka"), self._ratio, self._width
                ),
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
            if self._local_gripper is not None:
                return replace(self._state, gripper=self._local_gripper.get_state(busy=busy))
            return replace(
                self._state,
                gripper=GripperState(
                    self.config.get("gripper", {}).get("type", "franka"),
                    self._ratio,
                    self._width,
                    busy,
                ),
            )

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

    def _setup_gripper(self):
        kind = self.config.get("gripper", {}).get("type", "franka")
        self._local_gripper = None
        if kind == "dh5":
            from control._gripper import create_gripper_backend

            self._local_gripper = create_gripper_backend(self.config)
            self._local_gripper.connect()
        if kind != "franka":
            return
        from franka_msgs.action import Grasp, Move
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState
        from std_srvs.srv import Trigger

        prefix = self.cfg.get("gripper_node", "franka_gripper")
        self._action_types = {"move": Move, "grasp": Grasp}
        self._actions = {
            name: ActionClient(self._node, cls, self.topic(f"{prefix}/{name}"))
            for name, cls in self._action_types.items()
        }
        self._clients = list(self._actions.values())
        self._stop_client = self._node.create_client(Trigger, self.topic(f"{prefix}/stop"))
        self._trigger_type = Trigger
        self._gripper_sub = self._node.create_subscription(
            JointState, self.topic(f"{prefix}/joint_states"), self._on_gripper, 1
        )

    def _on_gripper(self, msg):
        if len(msg.position) == 2 and np.isfinite(msg.position).all():
            self._width = float(sum(msg.position))

    def _wait_future(self, future):
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(float(self.cfg.get("gripper_timeout_s", 10))):
            raise TimeoutError("Franka ROS gripper request timed out; verify device state")
        return future.result()

    def gripper_command(self, ratio, *, speed, force):
        if self._local_gripper is not None:
            return self._local_gripper.command(ratio, speed=speed, force=force)
        if self.config.get("gripper", {}).get("type", "franka") != "franka":
            raise RuntimeError("Gripper is disabled")
        name = "move" if ratio >= 0.5 else "grasp"
        client = self._actions[name]
        if not client.wait_for_server(timeout_sec=float(self.cfg.get("connect_timeout_s", 10))):
            raise TimeoutError(f"Franka gripper {name} action unavailable")
        goal = self._action_types[name].Goal()
        goal.width = (
            float(self.config.get("gripper", {}).get("open_width_m", 0.05)) * ratio
            if name == "move"
            else 0.0
        )
        goal.speed = float(speed)
        if name == "grasp":
            goal.force = float(force)
            goal.epsilon.inner = goal.epsilon.outer = 0.04
        handle = self._wait_future(client.send_goal_async(goal))
        if not handle.accepted:
            raise RuntimeError("Franka gripper goal rejected")
        try:
            result = self._wait_future(handle.get_result_async())
        except TimeoutError:
            handle.cancel_goal_async()
            raise
        from action_msgs.msg import GoalStatus

        success = result.status == GoalStatus.STATUS_SUCCEEDED and result.result.success
        if success:
            self._ratio = ratio
        return success

    def stop_gripper(self):
        if self.config.get("gripper", {}).get("type", "franka") == "none":
            return
        if self._local_gripper is not None:
            return self._local_gripper.stop()
        if not self._stop_client.wait_for_service(timeout_sec=2):
            raise TimeoutError("Franka gripper stop service unavailable")
        result = self._wait_future(self._stop_client.call_async(self._trigger_type.Request()))
        if not result.success:
            raise RuntimeError(result.message)

    def tactile_frames(self):
        return self._local_gripper.tactile_frames() if self._local_gripper else (None,) * 4

    def tactile_images(self):
        return self._local_gripper.tactile_images() if self._local_gripper else (None, None)

    def close(self):
        self._streaming = False
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=2)
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("ROS executor did not stop; resources retained")
        if getattr(self, "_local_gripper", None) is not None:
            self._local_gripper.close()
        for client in self._clients:
            client.destroy()
        self._clients.clear()
        if self._node is not None:
            self._node.destroy_node()
        if self._context is not None:
            self._context.try_shutdown()
            self._context.destroy()
        self._context = self._node = self._executor = self._thread = None
        self._state = self._failure = None
