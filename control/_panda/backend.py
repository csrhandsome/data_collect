"""Lazy SDK boundary. Connecting does not command motion."""

from __future__ import annotations

import os
import time

import numpy as np

from control._gripper import create_gripper_backend
from control.robotic_arm_types import RobotState
from control.util.pose import matrix_to_quat_xyzw


class PandaBackend:
    def __init__(self, config):
        self.config = config
        self.panda = self.desk = self.controller = None
        self.gripper = create_gripper_backend(config)

    def connect(self):
        import panda_py

        robot = self.config.get("robot", {})
        host = robot.get("hostname", "192.168.1.100")
        try:
            if robot.get("activate_fci", True):
                username = robot.get("username") or os.environ.get("FRANKA_USERNAME")
                password = robot.get("password") or os.environ.get("FRANKA_PASSWORD")
                if not username or not password:
                    raise ValueError("Set FRANKA_USERNAME and FRANKA_PASSWORD or robot credentials")
                self.desk = panda_py.Desk(host, username, password, platform="panda")
                self.desk.unlock()
                self.desk.activate_fci()
            self.panda = panda_py.Panda(host)
            self.gripper.connect()
        except BaseException:
            self.close()
            raise

    def close(self):
        errors = []
        for callback in [
            lambda: self.panda.stop_controller() if self.panda is not None else None,
            self.gripper.close,
            lambda: self.desk.deactivate_fci() if self.desk else None,
            lambda: self.desk.release_control() if self.desk else None,
            lambda: self.desk.logout() if self.desk else None,
        ]:
            try:
                callback()
            except Exception as exc:
                errors.append(exc)
        self.controller = self.panda = self.desk = None
        if errors:
            raise RuntimeError("Panda resource cleanup failed") from errors[0]

    def snapshot(self, busy=False):
        state = self.panda.get_state()
        transform = np.asarray(state.O_T_EE).reshape(4, 4).T
        return RobotState(
            np.asarray(state.q),
            np.asarray(state.dq),
            transform[:3, 3],
            matrix_to_quat_xyzw(transform[:3, :3]),
            transform,
            self.gripper.get_state(busy=busy),
            time.monotonic_ns(),
            robot_time_s=state.time.to_sec(),
            robot_mode=str(state.robot_mode),
            control_command_success_rate=float(state.control_command_success_rate),
            current_errors=str(state.current_errors),
        )

    def start(self, state):
        from panda_py import controllers

        tuning = self.config.get("control", {})
        controller = controllers.CartesianImpedance(
            filter_coeff=float(tuning.get("ee_filter_coeff", 0.35)),
            nullspace_stiffness=float(tuning.get("ee_nullspace_stiffness", 0.5)),
        )
        controller.set_control(state.ee_position, state.ee_quaternion_xyzw, state.joint_positions)
        self.panda.start_controller(controller)
        self.controller = controller

    def target(self, position, quaternion, nullspace):
        self.controller.set_control(position, quaternion, nullspace)

    def stop(self):
        self.panda.stop_controller()
        self.controller = None

    def move_joints(self, target, speed):
        return bool(self.panda.move_to_joint_position(target, speed_factor=speed))

    def move_ee(self, position, quaternion, speed):
        tuning = self.config.get("robot", {})
        return bool(
            self.panda.move_to_pose(
                position,
                quaternion,
                speed_factor=speed,
                success_threshold=float(tuning.get("position_tolerance_m", 0.01)),
                orientation_threshold=float(tuning.get("orientation_tolerance_rad", 0.05)),
            )
        )

    def cancel(self):
        self.panda.stop_controller()

    def pose_from_joints(self, joints):
        import panda_py

        return np.asarray(panda_py.fk(joints), dtype=np.float64)

    def gripper_command(self, ratio, *, speed, force):
        return self.gripper.command(ratio, speed=speed, force=force)

    def stop_gripper(self):
        self.gripper.stop()

    def tactile_frames(self):
        return self.gripper.tactile_frames()

    def tactile_images(self):
        return self.gripper.tactile_images()
