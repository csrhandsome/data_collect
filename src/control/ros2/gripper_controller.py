"""ROS action clients and state subscriptions for the selected gripper."""

import json
import logging
import threading
import time

import numpy as np

from control.types import GripperState

logger = logging.getLogger(__name__)


class Ros2GripperController:
    def __init__(self, config, node, *, source=None):
        self.config = config
        self.cfg = config.get("ros2", {})
        self._node = node
        self._source = source
        self._ratio, self._width = 1.0, None
        self._dh5_frames = {}
        self._dh5_ready = False
        self._clients = []
        self._reported_missing = False

    def get_state(self, *, busy=False):
        kind = self.config.get("gripper", {}).get("type", "franka")
        if self._franka_without_publisher():
            self._note_missing()
            return GripperState(kind, 0.0, 0.0, busy)
        return GripperState(kind, self._ratio, self._width, busy)

    def _franka_without_publisher(self):
        if self.config.get("gripper", {}).get("type", "franka") != "franka":
            return False
        subscription = getattr(self, "_gripper_sub", None)
        if subscription is None:
            return False
        return subscription.get_publisher_count() == 0

    def _note_missing(self):
        if self._reported_missing:
            return
        self._reported_missing = True
        logger.warning(
            "Franka gripper joint state is not subscribed; reporting closed (0) and skipping commands"
        )

    def connect(self):
        kind = self.config.get("gripper", {}).get("type", "franka")
        if kind == "dh5":
            self._connect_dh5()
        if kind != "franka":
            return
        from franka_msgs.action import Grasp, Move
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState
        from std_srvs.srv import Trigger

        prefix = self.cfg.get("gripper_node", "panda_gripper")
        self._action_types = {"move": Move, "grasp": Grasp}
        self._actions = {
            name: ActionClient(self._node, cls, self._node.resolve_topic_name(f"{prefix}/{name}"))
            for name, cls in self._action_types.items()
        }
        self._clients = list(self._actions.values())
        self._stop_client = self._node.create_client(Trigger, self._node.resolve_topic_name(f"{prefix}/stop"))
        self._trigger_type = Trigger
        subscribe = self._source.subscribe if self._source is not None else self._node.create_subscription
        self._gripper_sub = subscribe(
            JointState, self._node.resolve_topic_name(f"{prefix}/joint_states"), self._on_gripper, 1
        )

    def _on_gripper(self, msg):
        if len(msg.position) == 2 and np.isfinite(msg.position).all():
            self._width = float(sum(msg.position))
        else:
            return False

    def _connect_dh5(self):
        from franka_msgs.action import Move
        from rclpy.action import ActionClient
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        self._dh5_move = ActionClient(self._node, Move, "/dh5/move")
        self._clients.append(self._dh5_move)
        self._action_types = {"move": Move}
        self._dh5_message = String
        self._dh5_vr = self._node.create_publisher(String, "/dh5/runner_vr_command", 1)
        self._node.create_subscription(String, "/dh5/state", self._on_dh5_state, 1)
        if self.config.get("tactile", {}).get("enabled", False):
            for side in ("left", "right"):
                self._node.create_subscription(
                    Image, f"/dh5/{side}/image_raw",
                    lambda msg, side=side: self._on_dh5_image(side, msg), qos_profile_sensor_data,
                )

    def _on_dh5_state(self, message):
        try:
            ratio = json.loads(message.data)["target_position"]
        except (KeyError, ValueError, TypeError):
            return
        if isinstance(ratio, (float, int)) and 0 <= ratio <= 1000:
            self._ratio = ratio / 1000
            self._dh5_ready = True
            if self._source is not None:
                with self._source.condition:
                    self._source.condition.notify_all()

    def ready(self):
        return self.config.get("gripper", {}).get("type") != "dh5" or self._dh5_ready

    def _on_dh5_image(self, side, message):
        from control.ros2.camera_connector import image_rgb

        stamp = message.header.stamp
        source_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if source_ns <= 0:
            return
        try:
            frame = image_rgb(message)
        except ValueError as exc:
            logger.warning("Invalid DH5 %s image: %s", side, exc)
            return
        age = self._node.get_clock().now().nanoseconds - source_ns
        self._dh5_frames[side] = frame, time.monotonic_ns() - age

    def forward_vr(self, sample):
        self._dh5_vr.publish(self._dh5_message(data=json.dumps({
            "pose_monotonic_ns": sample.source_ns,
            "pose_seq": sample.source_seq,
            "gripper_velocity_axis": sample.axis,
        })))

    def _wait_future(self, future):
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(float(self.cfg.get("gripper_timeout_s", 10))):
            raise TimeoutError("Franka ROS gripper request timed out; verify device state")
        return future.result()

    def command(self, ratio, *, speed, force):
        kind = self.config.get("gripper", {}).get("type", "franka")
        if kind == "none":
            raise RuntimeError("Gripper is disabled")
        if self._franka_without_publisher():
            self._note_missing()
            return True
        name = "move" if kind == "dh5" or ratio >= 0.5 else "grasp"
        client = self._dh5_move if kind == "dh5" else self._actions[name]
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

    def stop(self):
        if self.config.get("gripper", {}).get("type", "franka") == "none":
            return
        if self.config.get("gripper", {}).get("type") == "dh5":
            raise NotImplementedError("DH5 driver has no confirmed stop command")
        if self._franka_without_publisher():
            self._note_missing()
            return
        if not self._stop_client.wait_for_service(timeout_sec=2):
            raise TimeoutError("Franka gripper stop service unavailable")
        result = self._wait_future(self._stop_client.call_async(self._trigger_type.Request()))
        if not result.success:
            raise RuntimeError(result.message)

    def tactile_frames(self):
        left, left_ns = self._dh5_frames.get("left", (None, None))
        right, right_ns = self._dh5_frames.get("right", (None, None))
        return left, right, left_ns, right_ns

    def tactile_images(self):
        left, right, _, _ = self.tactile_frames()
        return left, right

    def close(self):
        self._dh5_frames.clear()
        for client in self._clients:
            client.destroy()
        self._clients.clear()
