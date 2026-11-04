"""Workflow reset helpers exclusively using the public arm API."""

import logging
import sys

import numpy as np

from control.util.pose import as_vector

DEFAULT_START = [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]


def joint_count(config):
    return 6 if config.get("robot", {}).get("backend") == "ur_ros2" else 7


def start_joint_position(config):
    target = config.get("robot", {}).get("start_joint_position")
    if target is None and joint_count(config) == 6:
        raise ValueError("UR requires six configured start_joint_position values")
    return as_vector(
        target if target is not None else DEFAULT_START, joint_count(config), "start joints"
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


def handle_gripper(arm, sample, state, now_ns, last_axis_ns, config, *, on_command=None):
    if arm.gripper_busy or not arm.get_capabilities().gripper:
        return False, last_axis_ns
    cfg = config.get("gripper", {})
    ratio = None
    if sample.close or sample.open:
        ratio = float(sample.open)
    elif cfg.get("type") == "dh5":
        levels = int(cfg.get("discrete_level_count", 28))
        interval_ns = round(float(cfg.get("vr_step_interval_s", 0.08)) * 1e9)
        if (
            abs(sample.axis) > cfg.get("vr_axis_threshold", 0.55)
            and now_ns - last_axis_ns >= interval_ns
        ):
            ratio = float(
                np.clip(
                    state.gripper.commanded_open_ratio - np.sign(sample.axis) / (levels - 1), 0, 1
                )
            )
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
