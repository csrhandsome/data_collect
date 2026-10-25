"""Shared pose conversions. Quaternions use xyzw unless the name says wxyz."""

from __future__ import annotations

import math

import numpy as np


def as_vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite {size}D vector")
    return vector.copy()


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = as_vector(quat, 4, "xyzw quaternion")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("Quaternion must be nonzero")
    return quat / norm


def slerp_quat_xyzw(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate normalized xyzw quaternions along the shortest arc."""
    start, end = normalize_quat_xyzw(start), normalize_quat_xyzw(end)
    dot = float(np.dot(start, end))
    if dot < 0:
        end, dot = -end, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quat_xyzw(start + fraction * (end - start))
    angle = math.acos(dot)
    return (math.sin((1 - fraction) * angle) * start + math.sin(fraction * angle) * end) / math.sin(
        angle
    )

def nlerp_quat_xyzw(start, end, fraction):
    start, end = normalize_quat_xyzw(start), normalize_quat_xyzw(end)
    if np.dot(start, end) < 0:
        end = -end
    return normalize_quat_xyzw((1-fraction)*start+fraction*end)


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return as_vector(quat_xyzw, 4, "xyzw quaternion")[[3, 0, 1, 2]]


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return as_vector(quat_wxyz, 4, "wxyz quaternion")[[1, 2, 3, 0]]


def quat_angle_xyzw(a: np.ndarray, b: np.ndarray) -> float:
    dot = abs(float(np.dot(normalize_quat_xyzw(a), normalize_quat_xyzw(b))))
    return float(2.0 * math.acos(np.clip(dot, -1.0, 1.0)))


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("Rotation must be a finite 3x3 matrix")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        r = np.sqrt(1.0 + trace)
        w = 0.5 * r
        s = 0.5 / r
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        r = np.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k])
        values = [0.0, 0.0, 0.0]
        values[i] = 0.5 * r
        s = 0.5 / r
        w = (rotation[k, j] - rotation[j, k]) * s
        values[j] = (rotation[j, i] + rotation[i, j]) * s
        values[k] = (rotation[k, i] + rotation[i, k]) * s
        x, y, z = values
    return normalize_quat_xyzw(np.array([x, y, z, w], dtype=np.float64))


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("Rotation must be a finite 3x3 matrix")
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) < 1e-6:
        roll = 0.0
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    else:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def rpy_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = as_vector(rpy, 3, "RPY angles") / 2.0
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def position_quat_to_matrix(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_xyzw_to_matrix(quat_xyzw)
    matrix[:3, 3] = as_vector(position, 3, "EE position")
    return matrix


def matrix_to_pose6(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("EE transform must be a finite 4x4 matrix")
    return np.concatenate([matrix[:3, 3], matrix_to_rpy(matrix[:3, :3])])


def position_quat_to_pose6(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    return matrix_to_pose6(position_quat_to_matrix(position, quat_xyzw))


def pose6_to_quat_xyzw(pose: np.ndarray) -> np.ndarray:
    pose = as_vector(pose, 6, "EE pose")
    return rpy_to_quat_xyzw(pose[3:])
