"""Deterministic device substitute for dry runs and public-contract tests."""

import time

import numpy as np

from control.robotic_arm_types import GripperState, RobotState
from control.util.pose import position_quat_to_matrix


class FakeBackend:
    def __init__(self):
        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.position = np.array([0.4, 0.0, 0.4])
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        self.ratio = 1.0
        self.connected = self.streaming = False
        self.calls = []

    def connect(self):
        self.connected = True
        self.calls.append("connect")

    def close(self):
        self.connected = self.streaming = False
        self.calls.append("close")

    def snapshot(self, busy=False):
        return RobotState(
            self.q,
            self.dq,
            self.position,
            self.quaternion,
            position_quat_to_matrix(self.position, self.quaternion),
            GripperState("fake", self.ratio, 0.05 * self.ratio, busy),
            time.monotonic_ns(),
        )

    def start(self, state):
        self.streaming = True
        self.calls.append("start")

    def target(self, position, quaternion, nullspace):
        self.position = np.array(position, copy=True)
        self.quaternion = np.array(quaternion, copy=True)
        self.q[:3] = self.position
        self.calls.append("target")

    def stop(self):
        self.streaming = False
        self.calls.append("stop")

    def move_joints(self, target, speed):
        self.q = np.array(target, copy=True)
        self.calls.append("move_joints")
        return True

    def move_ee(self, position, quaternion, speed):
        self.position = np.array(position, copy=True)
        self.quaternion = np.array(quaternion, copy=True)
        self.calls.append("move_ee")
        return True

    def cancel(self):
        self.calls.append("cancel")

    def pose_from_joints(self, joints):
        return position_quat_to_matrix(np.asarray(joints)[:3], [0, 0, 0, 1])

    def gripper_command(self, ratio, *, speed, force):
        self.ratio = ratio
        self.calls.append("gripper")
        return True

    def stop_gripper(self):
        self.calls.append("stop_gripper")

    def tactile_images(self):
        return np.zeros((224, 224, 3), dtype=np.uint8), np.zeros((224, 224, 3), dtype=np.uint8)

    def tactile_frames(self):
        # A 30 Hz synthetic camera keeps repeated servo polls from inventing history.
        stamp = (time.monotonic_ns() // 33_333_333) * 33_333_333
        left, right = self.tactile_images()
        return left, right, stamp, stamp
