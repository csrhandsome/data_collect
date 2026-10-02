"""Workflow reset helpers exclusively using the public arm API."""

import logging
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
    arm.move_to_start()
    arm.start_stream()


def handle_gripper(arm, sample, state, now_ns, last_axis_ns, config):
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
