"""Shared robot poses, state, command results and camera payloads.

The learned 6D pose is [x, y, z, roll, pitch, yaw] in the robot base frame.
Angles use radians and the ZYX (yaw-pitch-roll) Euler convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from control.util.pose import (
    as_vector,
    matrix_to_pose6,
    pose6_to_quat_xyzw,
    position_quat_to_pose6,
)


def _with_gripper(arm_values: np.ndarray, gripper_open: float) -> np.ndarray:
    if not np.isfinite(gripper_open) or not 0.0 <= gripper_open <= 1.0:
        raise ValueError("Logical gripper state must be in 0..1")
    return np.concatenate([arm_values, [float(gripper_open)]]).astype(np.float32)


@dataclass(frozen=True)
class EEPose:
    """Base-frame xyz metres and ZYX roll/pitch/yaw radians."""

    xyz_m: np.ndarray
    rpy_rad: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyz_m", as_vector(self.xyz_m, 3, "EE xyz"))
        object.__setattr__(self, "rpy_rad", as_vector(self.rpy_rad, 3, "EE rpy"))
        self.xyz_m.setflags(write=False)
        self.rpy_rad.setflags(write=False)

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate([self.xyz_m, self.rpy_rad]).astype(np.float32)

    def action(self, gripper_open: float) -> np.ndarray:
        return _with_gripper(self.vector, gripper_open)

    def quaternion_xyzw(self) -> np.ndarray:
        return pose6_to_quat_xyzw(self.vector)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> EEPose:
        pose = matrix_to_pose6(matrix)
        return cls(pose[:3], pose[3:])

    @classmethod
    def from_position_quat(cls, position: np.ndarray, quat_xyzw: np.ndarray) -> EEPose:
        pose = position_quat_to_pose6(position, quat_xyzw)
        return cls(pose[:3], pose[3:])

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> EEPose:
        pose = as_vector(vector, 6, "EE pose")
        return cls(pose[:3], pose[3:])


@dataclass(frozen=True)
class JointPose:
    """Seven measured or commanded Franka joint angles in radians."""

    angles_rad: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "angles_rad", as_vector(self.angles_rad, 7, "Joint angles"))
        self.angles_rad.setflags(write=False)

    @property
    def vector(self) -> np.ndarray:
        return self.angles_rad.astype(np.float32)

    def action(self, gripper_open: float) -> np.ndarray:
        return _with_gripper(self.vector, gripper_open)


@dataclass(frozen=True)
class GripperState:
    kind: str
    commanded_open_ratio: float
    measured_width_m: float | None = None
    busy: bool = False


@dataclass(frozen=True)
class RobotState:
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    ee_position: np.ndarray
    ee_quaternion_xyzw: np.ndarray
    end_effector_pose: np.ndarray
    gripper: GripperState
    sampled_monotonic_ns: int
    base_frame: str = "panda_link0"
    robot_time_s: float | None = None
    robot_mode: str | None = None
    control_command_success_rate: float | None = None
    current_errors: str | None = None
    # External wrench at stiffness frame K, expressed in base_frame: N, N m.
    ee_wrench: np.ndarray | None = None

    def __post_init__(self):
        for name, shape in (
            ("joint_positions", (7,)),
            ("joint_velocities", (7,)),
            ("ee_position", (3,)),
            ("ee_quaternion_xyzw", (4,)),
            ("end_effector_pose", (4, 4)),
        ):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.ee_wrench is not None:
            value = as_vector(self.ee_wrench, 6, "EE external wrench")
            value.setflags(write=False)
            object.__setattr__(self, "ee_wrench", value)

    @property
    def ee_pose(self) -> EEPose:
        return EEPose.from_position_quat(self.ee_position, self.ee_quaternion_xyzw)


@dataclass(frozen=True)
class RobotCapabilities:
    ee_stream: bool = True
    joint_stream: bool = False
    finite_motion: bool = True
    gripper: bool = True
    tactile_images: bool = False


@dataclass(frozen=True)
class RobotStatus:
    connected: bool
    stream_space: Literal["ee"] | None
    motion_running: bool
    gripper_busy: bool


@dataclass(frozen=True)
class EEStreamTarget:
    """Last locally accepted EE target, including gripper pause/resume holds."""

    ee_position: np.ndarray
    ee_quaternion_xyzw: np.ndarray
    submitted_monotonic_ns: int
    sampled_monotonic_ns: int
    phase: Literal["start", "command", "hold", "gripper_pause", "gripper_resume"]

    def __post_init__(self):
        for name, shape in (("ee_position", (3,)), ("ee_quaternion_xyzw", (4,))):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CommandReceipt:
    """Local acceptance, not confirmation that a target was reached."""

    space: Literal["ee", "gripper"]
    submitted_monotonic_ns: int
    accepted: bool = True


@dataclass(frozen=True)
class MotionResult:
    status: Literal["succeeded", "cancelled", "idle"]
    duration_s: float
    final_state: RobotState


@dataclass(frozen=True)
class CameraPair:
    front: np.ndarray
    wrist: np.ndarray
    front_ns: int
    wrist_ns: int
    front_sensor_timestamp: float
    wrist_sensor_timestamp: float
