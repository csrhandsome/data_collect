"""Causal latest-sample resampling. Button edges never repeat at the servo rate."""

from dataclasses import dataclass

import numpy as np

from control.util.pose import (
    matrix_to_quat_xyzw,
    normalize_quat_xyzw,
    quat_xyzw_to_matrix,
)


@dataclass(frozen=True)
class ResampledVR:
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    enabled: bool
    save: bool
    discard: bool
    close: bool
    open: bool
    axis: float
    source_ns: int
    source_seq: int


class VRResampler:
    def __init__(self, stale_after_s=0.5):
        self.stale_ns = round(stale_after_s * 1e9)
        self.previous = {}

    def sample(self, vr, now_ns):
        source_ns = int(vr.pose_monotonic_ns)
        fresh = 0 <= now_ns - source_ns < self.stale_ns and source_ns > 0
        edges = []
        for name in ["y_pressed", "x_pressed", "gripper_close", "gripper_open"]:
            value = bool(getattr(vr, name, False)) and fresh
            edges.append(value and not self.previous.get(name, False))
            self.previous[name] = value
        return ResampledVR(
            np.array([vr.pos_x, vr.pos_y, vr.pos_z]),
            normalize_quat_xyzw([vr.quat_x, vr.quat_y, vr.quat_z, vr.quat_w]),
            bool(vr.arm_enabled and fresh),
            *edges,
            float(vr.gripper_velocity_axis),
            source_ns,
            int(vr.pose_seq),
        )


class EEMapper:
    """Anchored VR mapping with velocity limits and a latched idle pose."""

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self, state=None):
        """Clear the VR anchor and optionally seed the pose held while idle."""
        self.anchor = None
        self.last_ns = None
        self._hold_pose = (
            (state.ee_position.copy(), state.ee_quaternion_xyzw.copy())
            if state is not None
            else None
        )
        self._command_pose = None
        self._desired_pose = self._hold_pose
        self._translation_limited = self._rotation_limited = False
        self._workspace_limited = False
        self._dt_s = 0.0

    def diagnostic_state(self):
        """Describe the unthrottled goal separately from the rate-limited command."""
        position, quaternion = (
            self._desired_pose if self._desired_pose is not None else (None, None)
        )
        return {
            "desired_position": position.tolist() if position is not None else None,
            "desired_quaternion_xyzw": quaternion.tolist() if quaternion is not None else None,
            "translation_speed_limited": self._translation_limited,
            "rotation_speed_limited": self._rotation_limited,
            "workspace_limited": self._workspace_limited,
            "dt_s": self._dt_s,
        }

    def map(self, sample, state, now_ns):
        if not sample.enabled:
            # Capture once at startup, trigger release, or loss of fresh VR input.
            # Following the measured pose every tick would allow gravity-induced drift.
            if self.anchor is not None or self._hold_pose is None:
                self.reset(state)
            position, quaternion = self._hold_pose
            return position.copy(), quaternion.copy()
        if self.anchor is None:
            self.anchor = (
                sample.position.copy(),
                quat_xyzw_to_matrix(sample.quaternion_xyzw),
                state.ee_position.copy(),
                quat_xyzw_to_matrix(state.ee_quaternion_xyzw),
            )
            self._command_pose = (state.ee_position.copy(), state.ee_quaternion_xyzw.copy())
        vr_pos, vr_rot, robot_pos, robot_rot = self.anchor
        from scipy.spatial.transform import Rotation

        gain = float(self.config.get("translation_gain", 1.0))
        offset = (sample.position - vr_pos) * gain
        limit = float(self.config.get("translation_limit_m", 0.5))
        clipped_offset = np.clip(offset, -limit, limit)
        self._workspace_limited = bool(np.any(offset != clipped_offset))
        offset = clipped_offset
        desired = robot_pos + offset
        # Rate-limit the command trajectory, not its error relative to the robot.
        # Otherwise delayed feedback pins the target a few millimetres ahead forever.
        command_position, command_quaternion = self._command_pose
        delta = desired - command_position
        dt = 0.01 if self.last_ns is None else max(0, (now_ns - self.last_ns) / 1e9)
        self.last_ns = now_ns
        self._dt_s = dt
        max_step = float(self.config.get("max_translation_speed_m_s", 0.2)) * dt
        norm = np.linalg.norm(delta)
        self._translation_limited = bool(norm > max_step)
        position = command_position + delta * min(1, max_step / max(norm, 1e-12))
        target_rot = robot_rot
        desired_rot = robot_rot
        self._rotation_limited = False
        if self.config.get("enable_rotation", True):
            relative = Rotation.from_matrix(
                quat_xyzw_to_matrix(sample.quaternion_xyzw) @ vr_rot.T
            ).as_rotvec()
            relative *= float(self.config.get("rotation_gain", 0.4))
            limit = float(self.config.get("rotation_limit_rad", 1.2))
            self._workspace_limited |= bool(np.linalg.norm(relative) > limit)
            relative *= min(1, limit / max(np.linalg.norm(relative), 1e-12))
            desired_rot = Rotation.from_rotvec(relative).as_matrix() @ robot_rot
            current = quat_xyzw_to_matrix(command_quaternion)
            error = Rotation.from_matrix(desired_rot @ current.T).as_rotvec()
            max_angle = float(self.config.get("max_rotation_speed_rad_s", 0.2)) * dt
            self._rotation_limited = bool(np.linalg.norm(error) > max_angle)
            error *= min(1, max_angle / max(np.linalg.norm(error), 1e-12))
            target_rot = Rotation.from_rotvec(error).as_matrix() @ current
        quaternion = matrix_to_quat_xyzw(target_rot)
        self._desired_pose = (desired, matrix_to_quat_xyzw(desired_rot))
        self._command_pose = (position.copy(), quaternion.copy())
        return position, quaternion
