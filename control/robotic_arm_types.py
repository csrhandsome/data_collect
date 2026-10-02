"""Public contracts: metres, radians, base-frame poses, xyzw quaternions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from control.robot_state import EEPose


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
