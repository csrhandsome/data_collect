"""Lazy SDK boundary. Connecting does not command motion."""

from __future__ import annotations

import os
import time

import numpy as np

from control.robotic_arm_types import GripperState, RobotState
from control.util.pose import matrix_to_quat_xyzw


class PandaBackend:
    def __init__(self, config):
        self.config = config
        self.panda = self.desk = self.gripper = self.controller = None
        self.gripper_kind = config.get("gripper", {}).get("type", "franka")
        self.commanded_open_ratio = 1.0
        self._width = None

    def connect(self):
        import panda_py
        from panda_py import libfranka

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
            config = self.config.get("gripper", {})
            if self.gripper_kind == "franka":
                self.gripper = libfranka.Gripper(host)
                self._width = float(self.gripper.read_once().width)
            elif self.gripper_kind == "dh5":
                from control.soft_gripper_control import DH5Gripper

                self.gripper = DH5Gripper(
                    **config.get("dh5", {}),
                    enable_cameras=self.config.get("tactile", {}).get("enabled", False),
                )
                self.gripper.set_force(int(config.get("force", 50)))
                self.gripper.set_velocity(int(config.get("velocity", 100)))
            elif self.gripper_kind != "none":
                raise ValueError(f"Unknown gripper: {self.gripper_kind}")
        except BaseException:
            self.close()
            raise

    def close(self):
        errors = []
        for callback in [
            lambda: self.panda.stop_controller() if self.panda is not None else None,
            lambda: self.gripper.close() if self.gripper_kind == "dh5" and self.gripper else None,
            lambda: self.desk.deactivate_fci() if self.desk else None,
            lambda: self.desk.release_control() if self.desk else None,
            lambda: self.desk.logout() if self.desk else None,
        ]:
            try:
                callback()
            except Exception as exc:
                errors.append(exc)
        self.controller = self.panda = self.gripper = self.desk = None
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
            GripperState(self.gripper_kind, self.commanded_open_ratio, self._width, busy),
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
        if self.gripper_kind == "none":
            raise RuntimeError("Gripper is disabled")
        if self.gripper_kind == "dh5":
            self.gripper.set_position(round((1 - ratio) * 1000), wait=True)
            result = True
        elif ratio >= 0.5:
            width = float(self.config.get("gripper", {}).get("open_width_m", 0.05)) * ratio
            result = self.gripper.move(width, speed)
        else:
            result = self.gripper.grasp(0.0, speed, force, 0.04, 0.04)
        if result:
            self.commanded_open_ratio = ratio
            if self.gripper_kind == "franka":
                self._width = float(self.gripper.read_once().width)
        return bool(result)

    def stop_gripper(self):
        if self.gripper_kind == "franka" and self.gripper is not None:
            self.gripper.stop()
        elif self.gripper_kind == "dh5":
            raise NotImplementedError("DH5 driver has no confirmed stop command")

    def tactile_images(self):
        if self.gripper_kind != "dh5":
            return None, None
        observation = self.gripper.observation
        return observation.wrist_img, observation.external_img
