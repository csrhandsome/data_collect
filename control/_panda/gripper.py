"""One owned gripper worker; workflows do not create gripper threads."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from control._panda.lifecycle import require_connected
from control.robotic_arm_types import CommandReceipt


def command(arm, ratio, speed, force, wait, timeout):
    require_connected(arm)
    if not arm.gripper.enabled:
        raise RuntimeError("Gripper is disabled")
    if (
        not np.isfinite([ratio, speed, force]).all()
        or not 0 <= ratio <= 1
        or speed <= 0
        or force <= 0
    ):
        raise ValueError("Gripper ratio must be 0..1, speed and force positive")
    with arm._lock:
        if arm.gripper_busy or arm._motion_running:
            raise RuntimeError("Gripper or finite motion is busy")
        if arm._gripper_executor is None:
            arm._gripper_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gripper")
        arm._gripper_future = arm._gripper_executor.submit(execute, arm, ratio, speed, force)
        future = arm._gripper_future
    if wait and not future.result(timeout=timeout):
        raise RuntimeError("Gripper command failed")
    return CommandReceipt("gripper", time.monotonic_ns())


def execute(arm, ratio, speed, force):
    with arm._lock:
        resume = arm._streaming
        if resume:
            state = arm.get_state()
            arm._backend.target(state.ee_position, state.ee_quaternion_xyzw, state.joint_positions)
            arm._backend.stop()
    try:
        return arm._backend.gripper_command(ratio, speed=speed, force=force)
    finally:
        with arm._lock:
            if resume:
                state = arm._backend.snapshot(busy=True)
                arm._backend.start(state)
                arm._nullspace = state.joint_positions.copy()


def wait(arm, timeout):
    future = arm._gripper_future
    if future is not None and not future.result(timeout=timeout):
        raise RuntimeError("Gripper command failed")
