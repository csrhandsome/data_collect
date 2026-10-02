"""Validated EE streaming and finite moves; trajectory times are seconds."""

import time

import numpy as np

from control._panda.lifecycle import require_connected
from control.robotic_arm_types import CommandReceipt, MotionResult
from control.util.pose import as_vector, normalize_quat_xyzw, quat_angle_xyzw


def send_ee(arm, position, quaternion):
    position = as_vector(position, 3, "EE position")
    quaternion = normalize_quat_xyzw(quaternion)
    require_connected(arm)
    with arm._lock:
        if not arm._streaming:
            raise RuntimeError("EE stream is not started")
        if arm.gripper_busy:
            raise RuntimeError("Gripper command owns the stream")
        limits = arm._config.get("control", {})
        state = arm.get_state()
        if np.linalg.norm(position - state.ee_position) > float(
            limits.get("max_target_translation_m", 0.15)
        ):
            raise ValueError("EE target exceeds measured translation guard")
        if quat_angle_xyzw(quaternion, state.ee_quaternion_xyzw) > float(
            limits.get("max_target_rotation_rad", 0.5)
        ):
            raise ValueError("EE target exceeds measured rotation guard")
        arm._backend.target(position, quaternion, arm._nullspace)
    return CommandReceipt("ee", time.monotonic_ns())


def hold(arm):
    require_connected(arm)
    with arm._lock:
        if arm.gripper_busy:
            raise RuntimeError("Gripper command owns the stream")
        if not arm._streaming:
            raise RuntimeError("EE stream is not started")
        state = arm.get_state()
        arm._nullspace = state.joint_positions.copy()
        arm._backend.target(state.ee_position, state.ee_quaternion_xyzw, arm._nullspace)
        return state


def begin(arm):
    require_connected(arm)
    with arm._lock:
        if arm._streaming or arm._motion_running or arm.gripper_busy:
            raise RuntimeError("Stop streaming and wait for gripper before finite motion")
        arm._motion_running = True
        arm._cancel_event.clear()
        arm._motion_done.clear()


def move(arm, target, quaternion=None, *, speed_factor=0.2):
    target = as_vector(target, 7 if quaternion is None else 3, "motion target")
    quaternion = normalize_quat_xyzw(quaternion) if quaternion is not None else None
    if not np.isfinite(speed_factor) or not 0 < speed_factor <= 1:
        raise ValueError("speed_factor must be in (0, 1]")
    begin(arm)
    started = time.monotonic()
    try:
        result = (
            arm._backend.move_joints(target, speed_factor)
            if quaternion is None
            else arm._backend.move_ee(target, quaternion, speed_factor)
        )
        if not result and not arm._cancel_event.is_set():
            raise RuntimeError("Backend did not confirm motion success")
        arm.wait_until_stopped()
        return MotionResult(
            "cancelled" if arm._cancel_event.is_set() else "succeeded",
            time.monotonic() - started,
            arm.get_state(),
        )
    finally:
        arm._motion_running = False
        arm._motion_done.set()


def trajectory(arm, points, on_sample):
    from control.util.pose import slerp_quat_xyzw

    points = list(points)
    if len(points) < 2:
        raise ValueError("EE trajectory needs at least two points")
    times = np.asarray([point["time_s"] for point in points], dtype=float)
    positions = [as_vector(point["position"], 3, "position") for point in points]
    quats = [normalize_quat_xyzw(point["quaternion_xyzw"]) for point in points]
    if not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory times must start at zero and strictly increase")
    begin(arm)
    started = time.monotonic()
    try:
        state = arm.get_state()
        arm._backend.start(state)
        while not arm._cancel_event.is_set():
            elapsed = min(time.monotonic() - started, times[-1])
            i = min(np.searchsorted(times, elapsed, side="right") - 1, len(points) - 2)
            fraction = (elapsed - times[i]) / (times[i + 1] - times[i])
            position = positions[i] + fraction * (positions[i + 1] - positions[i])
            quaternion = slerp_quat_xyzw(quats[i], quats[i + 1], fraction)
            measured = arm.get_state()
            limits = arm._config.get("control", {})
            if np.linalg.norm(position - measured.ee_position) > limits.get(
                "max_target_translation_m", 0.15
            ) or quat_angle_xyzw(quaternion, measured.ee_quaternion_xyzw) > limits.get(
                "max_target_rotation_rad", 0.5
            ):
                raise ValueError("Trajectory exceeds measured pose guard")
            arm._backend.target(position, quaternion, state.joint_positions)
            if on_sample:
                on_sample(arm.get_state())
            if elapsed >= times[-1]:
                break
            arm._cancel_event.wait(0.01)
        if not arm._cancel_event.is_set():
            deadline = time.monotonic() + float(
                arm._config.get("robot", {}).get("arrival_timeout_s", 2)
            )
            while True:
                measured = arm.get_state()
                tuning = arm._config.get("robot", {})
                if np.linalg.norm(measured.ee_position - positions[-1]) <= tuning.get(
                    "position_tolerance_m", 0.01
                ) and quat_angle_xyzw(measured.ee_quaternion_xyzw, quats[-1]) <= tuning.get(
                    "orientation_tolerance_rad", 0.05
                ):
                    break
                if arm._cancel_event.is_set():
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Trajectory endpoint was not measured within arrival tolerance"
                    )
                arm._cancel_event.wait(0.01)
        arm._backend.stop()
        arm.wait_until_stopped()
        return MotionResult(
            "cancelled" if arm._cancel_event.is_set() else "succeeded",
            time.monotonic() - started,
            arm.get_state(),
        )
    finally:
        try:
            arm._backend.stop()
        finally:
            arm._motion_running = False
            arm._motion_done.set()


def cancel(arm, timeout_s):
    require_connected(arm)
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Cancellation timeout must be positive")
    if not arm._motion_running:
        return MotionResult("idle", 0.0, arm.get_state())
    arm._cancel_event.set()
    arm._backend.cancel()
    if not arm._motion_done.wait(timeout_s):
        raise TimeoutError("Cancellation requested; completion unconfirmed")
    return MotionResult("cancelled", 0.0, arm.get_state())
