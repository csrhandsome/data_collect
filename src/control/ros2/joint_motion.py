"""Bounded joint reset over the existing ROS impedance controller."""

import logging
import time

import numpy as np

from control.util.pose import as_vector

JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
logger = logging.getLogger(__name__)


def validate_joint_target(target):
    target = as_vector(target, 7, "start joints")
    if np.any(target <= JOINT_LOWER) or np.any(target >= JOINT_UPPER):
        raise ValueError("Joint reset target must lie inside Panda joint limits")
    return target


def move_joints(backend, target, speed):
    target = validate_joint_target(target)
    if not np.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("speed_factor must be in (0, 1]")
    if backend._streaming:
        raise RuntimeError("Stop the EE stream before joint reset")
    try:
        backend._source.wait(
            lambda: backend._joint_publisher.get_subscription_count() > 0,
            float(backend.cfg.get("connect_timeout_s", 10)),
        )
    except TimeoutError as exc:
        raise RuntimeError("No joint reset subscriber; rebuild the ROS controller image") from exc
    robot = backend.config.get("robot", {})
    start = backend.snapshot().joint_positions.copy()
    distance = float(np.max(np.abs(target - start)))
    # Quintic interpolation has peak derivative 1.875 and acceleration < 5.774.
    duration = max(2.0, 1.875 * distance / speed, np.sqrt(5.774 * distance / speed))
    arrival_timeout = float(robot.get("arrival_timeout_s", 5.0))
    tolerance = float(robot.get("joint_tolerance_rad", 0.02))
    tracking_limit = float(robot.get("max_reset_joint_error_rad", 0.2))
    backend._joint_cancel.clear()
    started = time.monotonic()
    settled_since = None
    logger.info("Joint reset: max displacement %.4f rad, trajectory %.3f s", distance, duration)
    backend._streaming = True  # Allows the measured Cartesian hold in finally.
    try:
        while not backend._joint_cancel.is_set():
            elapsed = time.monotonic() - started
            fraction = min(elapsed / duration, 1.0)
            blend = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
            desired = start + blend * (target - start)
            state = backend.snapshot()
            if np.max(np.abs(desired - state.joint_positions)) > tracking_limit:
                raise ValueError("Joint reset exceeds measured joint tracking guard")
            if backend._joint_publisher.get_subscription_count() == 0:
                raise RuntimeError("Joint reset controller subscriber disappeared")
            backend.publish_joints(desired)
            arrived = (
                fraction == 1.0
                and np.max(np.abs(target - state.joint_positions)) <= tolerance
                and np.max(np.abs(state.joint_velocities)) <= 0.005
            )
            if arrived:
                settled_since = time.monotonic() if settled_since is None else settled_since
                if time.monotonic() - settled_since >= 0.2:
                    return True
            else:
                settled_since = None
            if elapsed >= duration + arrival_timeout:
                error = target - state.joint_positions
                raise TimeoutError(
                    "Joint reset endpoint was not measured within arrival tolerance: "
                    f"elapsed={elapsed:.3f} s, trajectory={duration:.3f} s, "
                    f"position_tolerance={tolerance:.6f} rad, velocity_tolerance=0.005 rad/s, "
                    f"max_position_error={np.max(np.abs(error)):.6f} rad, "
                    f"max_velocity={np.max(np.abs(state.joint_velocities)):.6f} rad/s, "
                    f"target_q={target.tolist()}, measured_q={state.joint_positions.tolist()}, "
                    f"error_q={error.tolist()}, measured_dq={state.joint_velocities.tolist()}"
                )
            backend._joint_cancel.wait(0.01)
        return False
    finally:
        try:
            # A fresh measured pose cancels joint mode in the realtime controller.
            # If feedback is unavailable its joint-command watchdog stops progression.
            state = backend.snapshot()
            backend.target(state.ee_position, state.ee_quaternion_xyzw, None)
        finally:
            backend._streaming = False
