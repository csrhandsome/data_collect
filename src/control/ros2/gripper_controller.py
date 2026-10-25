"""Franka ROS action clients and gripper state subscription; DH5 stays local."""

import threading

import numpy as np

from control.types import GripperState


class Ros2GripperController:
    def __init__(self, config, node):
        self.config = config
        self.cfg = config.get("ros2", {})
        self._node = node
        self._ratio, self._width = 1.0, None
        self._local_gripper = None
        self._clients = []

    def get_state(self, *, busy=False):
        if self._local_gripper is not None:
            return self._local_gripper.get_state(busy=busy)
        return GripperState(self.config.get("gripper", {}).get("type", "franka"),
                            self._ratio, self._width, busy)

    def connect(self):
        kind = self.config.get("gripper", {}).get("type", "franka")
        self._local_gripper = None
        if kind == "dh5":
            from control.direct.grippers import create_gripper_backend

            self._local_gripper = create_gripper_backend(self.config)
            self._local_gripper.connect()
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
        self._gripper_sub = self._node.create_subscription(
            JointState, self._node.resolve_topic_name(f"{prefix}/joint_states"), self._on_gripper, 1
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

    def command(self, ratio, *, speed, force):
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

    def stop(self):
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
        if self._local_gripper is not None:
            self._local_gripper.close()
        for client in self._clients:
            client.destroy()
        self._clients.clear()
