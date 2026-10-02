"""Robot arm state representations shared by collection and inference.

The learned 6D pose is [x, y, z, roll, pitch, yaw] in the robot base frame.
Angles use radians and the ZYX (yaw-pitch-roll) Euler convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.util.pose import (
    as_vector,
    matrix_to_pose6,
    pose6_to_quat_xyzw,
    position_quat_to_pose6,
    quat_angle_xyzw,
)

__all__ = [
    "EEPose",
    "JointPose",
    "matrix_to_pose6",
    "pose6_to_quat_xyzw",
    "position_quat_to_pose6",
    "quat_angle_xyzw",
]


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
