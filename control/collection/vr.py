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
    """Absolute anchored mapping with velocity limits in metres/s and rad/s."""

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.anchor = None
        self.last_ns = None

    def map(self, sample, state, now_ns):
        if not sample.enabled:
            self.reset()
            return state.ee_position.copy(), state.ee_quaternion_xyzw.copy()
        if self.anchor is None:
            self.anchor = (
                sample.position.copy(),
                quat_xyzw_to_matrix(sample.quaternion_xyzw),
                state.ee_position.copy(),
                quat_xyzw_to_matrix(state.ee_quaternion_xyzw),
            )
        vr_pos, vr_rot, robot_pos, robot_rot = self.anchor
        from scipy.spatial.transform import Rotation

        gain = float(self.config.get("translation_gain", 1.0))
        offset = (sample.position - vr_pos) * gain
        limit = float(self.config.get("translation_limit_m", 0.5))
        offset = np.clip(offset, -limit, limit)
        desired = robot_pos + offset
        delta = desired - state.ee_position
        dt = 0.01 if self.last_ns is None else max(0, (now_ns - self.last_ns) / 1e9)
        self.last_ns = now_ns
        max_step = float(self.config.get("max_translation_speed_m_s", 0.2)) * dt
        norm = np.linalg.norm(delta)
        position = state.ee_position + delta * min(1, max_step / max(norm, 1e-12))
        target_rot = robot_rot
        if self.config.get("enable_rotation", True):
            relative = Rotation.from_matrix(
                quat_xyzw_to_matrix(sample.quaternion_xyzw) @ vr_rot.T
            ).as_rotvec()
            relative *= float(self.config.get("rotation_gain", 0.4))
            limit = float(self.config.get("rotation_limit_rad", 1.2))
            relative *= min(1, limit / max(np.linalg.norm(relative), 1e-12))
            desired_rot = Rotation.from_rotvec(relative).as_matrix() @ robot_rot
            current = quat_xyzw_to_matrix(state.ee_quaternion_xyzw)
            error = Rotation.from_matrix(desired_rot @ current.T).as_rotvec()
            max_angle = float(self.config.get("max_rotation_speed_rad_s", 0.2)) * dt
            error *= min(1, max_angle / max(np.linalg.norm(error), 1e-12))
            target_rot = Rotation.from_rotvec(error).as_matrix() @ current
        return position, matrix_to_quat_xyzw(target_rot)
