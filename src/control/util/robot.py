"""Workflow reset helpers exclusively using the public arm API."""

import logging
import math
import sys

import numpy as np

from control.util.pose import as_vector

DEFAULT_START = [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]


def start_joint_position(config):
    return as_vector(
        config.get("robot", {}).get("start_joint_position") or DEFAULT_START, 7, "start joints"
    )


def reset_robot(arm):
    arm.wait_gripper(timeout=5.0)
    arm.stop_stream()
    if arm.get_capabilities().gripper:
        arm.gripper_open()
    if arm._config.get("robot", {}).get("move_to_start", True):
        logging.getLogger(__name__).info("Returning robot to configured start joints")
        result = arm.move_to_start()
        if result.status != "succeeded":
            raise RuntimeError("Return to start pose was not completed")
        logging.getLogger(__name__).info("Robot returned to start joints")
    arm.start_stream()


DEFAULT_GRIPPER_AXIS_DEADZONE = 0.10


def dh5_vr_position_step(axis, deadzone, levels):
    """Return the signed increment in DH5 position units for one update."""
    if not math.isfinite(axis) or not 0 <= deadzone < 1 or levels <= 1:
        return 0
    magnitude = max(0.0, (min(abs(axis), 1.0) - deadzone) / (1.0 - deadzone))
    return round(math.copysign(magnitude * 1000 / (levels - 1), axis))


def handle_gripper(arm, sample, state, now_ns, last_axis_ns, config, *, on_command=None):
    if arm.gripper_busy or not arm.get_capabilities().gripper:
        return False, last_axis_ns
    cfg = config.get("gripper", {})
    ratio = None
    if sample.close or sample.open:
        ratio = float(sample.open)
    elif cfg.get("type") == "dh5":
        deadzone = float(cfg.get("vr_axis_threshold", DEFAULT_GRIPPER_AXIS_DEADZONE))
        gripper = getattr(arm, "gripper", None)
        if gripper is not None and gripper.forward_vr(sample):
            active = abs(sample.axis) > deadzone
            if active and on_command is not None:
                on_command(state.gripper.commanded_open_ratio)
            return active, last_axis_ns
        levels = int(cfg.get("discrete_level_count", 28))
        interval_ns = round(float(cfg.get("vr_step_interval_s", 0.08)) * 1e9)
        if now_ns - last_axis_ns >= interval_ns:
            step = dh5_vr_position_step(sample.axis, deadzone, levels)
            if step:
                current = state.gripper.commanded_open_ratio
                target = min(1.0, max(0.0, current - step / 1000))
                if target != current:
                    ratio = target
                    last_axis_ns = now_ns
    if ratio is None:
        return False, last_axis_ns
    arm.set_gripper(
        ratio, speed=float(cfg.get("speed_m_s", 0.2)), force=float(cfg.get("force", 50)), wait=False
    )
    if on_command is not None:
        on_command(ratio)
    return True, last_axis_ns


def finish_stream(arm):
    """Preserve an active workflow exception while attempting controller cleanup."""
    failed = sys.exc_info()[0] is not None
    try:
        arm.wait_gripper(timeout=5)
        if arm.get_status().stream_space is not None:
            arm.hold()
            arm.stop_stream()
    except Exception:
        if not failed:
            raise
        logging.getLogger(__name__).exception("Stream cleanup failed during workflow failure")
