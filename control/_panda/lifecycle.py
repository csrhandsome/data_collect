"""Controller lifecycle and measured stop confirmation."""

import threading
import time

import numpy as np

from control.robotic_arm_types import RobotStatus


def require_connected(arm):
    if not arm._connected:
        raise RuntimeError("Robot is disconnected; call connect() first")


def connect(arm):
    if not arm._connected:
        arm._backend.connect()
        arm._connected = True
    return status(arm)


def status(arm):
    return RobotStatus(
        arm._connected,
        "ee" if arm._streaming else None,
        arm._motion_running,
        arm._gripper_future is not None and not arm._gripper_future.done(),
    )


def wait_stopped(arm, timeout_s=2.0, velocity_tol=5e-3):
    require_connected(arm)
    if not np.isfinite(timeout_s) or timeout_s <= 0 or velocity_tol <= 0:
        raise ValueError("Stop timeout and velocity tolerance must be positive")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = arm.get_state()
        if np.max(np.abs(state.joint_velocities)) <= velocity_tol:
            return state
        time.sleep(0.005)
    raise TimeoutError("Measured joint velocities did not settle")


def start_stream(arm):
    require_connected(arm)
    with arm._lock:
        if arm._motion_running or arm.gripper_busy:
            raise RuntimeError("Finite motion or gripper is running")
        if not arm._streaming:
            state = arm.get_state()
            arm._backend.start(state)
            arm._streaming = True
            arm._nullspace = state.joint_positions.copy()


def stop_stream(arm, timeout_s):
    require_connected(arm)
    with arm._lock:
        if arm.gripper_busy:
            raise RuntimeError("Wait for gripper before stopping the stream")
        if arm._streaming:
            arm.hold()
            arm._backend.stop()
            wait_stopped(arm, timeout_s)
            arm._streaming = False


def close(arm):
    if not arm._connected:
        return
    errors = []
    for callback in [
        lambda: arm.cancel_motion(),
        lambda: arm.wait_gripper(timeout=2.0),
        lambda: stop_stream(arm, 2.0),
    ]:
        try:
            callback()
        except Exception as exc:
            errors.append(exc)
    if arm.gripper_busy:
        raise RuntimeError("Gripper worker still running; resources retained") from errors[0]
    try:
        arm._backend.close()
    finally:
        arm._connected = arm._streaming = False
        if arm._gripper_executor is not None:
            arm._gripper_executor.shutdown(wait=True)
            arm._gripper_executor = None
    if errors:
        raise RuntimeError("Robot cleanup encountered a failure") from errors[0]


def initialize(arm, config, backend):
    from control._panda.backend import PandaBackend

    arm._config = config or {}
    arm._backend = backend if backend is not None else PandaBackend(arm._config)
    arm._connected = arm._streaming = arm._motion_running = False
    arm._lock = threading.RLock()
    arm._cancel_event = threading.Event()
    arm._motion_done = threading.Event()
    arm._motion_done.set()
    arm._nullspace = None
    arm._gripper_executor = arm._gripper_future = None
