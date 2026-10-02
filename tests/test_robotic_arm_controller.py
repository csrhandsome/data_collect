"""Every public controller interface, with command ownership and failure checks."""

import ast
import threading
import time
from dataclasses import is_dataclass
from pathlib import Path

import numpy as np
import pytest

from control._panda.fake import FakeBackend
from control.robotic_arm_controller import RoboticArmControler


@pytest.fixture
def arm():
    robot = RoboticArmControler(backend=FakeBackend())
    robot.connect()
    yield robot
    robot.close()


def test_lifecycle_context_and_status():
    backend = FakeBackend()
    arm = RoboticArmControler(backend=backend)
    assert backend.calls == []
    assert not arm.get_status().connected
    assert arm.get_capabilities().ee_stream and not arm.get_capabilities().joint_stream
    arm.close()
    with pytest.raises(RuntimeError):
        arm.get_state()
    assert arm.__enter__() is arm
    assert arm.connect().connected
    assert backend.calls.count("connect") == 1
    arm.__exit__(None, None, None)
    arm.close()
    assert not backend.connected
    with RoboticArmControler(backend=FakeBackend()) as other:
        assert other.get_status().connected


def test_snapshot_is_immutable_and_ready(arm):
    state = arm.get_state()
    assert is_dataclass(state) and is_dataclass(state.gripper)
    arm._backend.q[:] = 1
    assert np.all(state.joint_positions == 0)
    with pytest.raises(ValueError):
        state.joint_positions[:] = 1
    assert arm.wait_ready(timeout_s=0.1).joint_positions.shape == (7,)
    assert arm.wait_until_stopped(timeout_s=0.1).ee_pose.vector.shape == (6,)


def test_ee_stream_and_hold(arm):
    with pytest.raises(ValueError, match="EE-only"):
        arm.start_stream("joint")
    with pytest.raises(RuntimeError):
        arm.send_ee_target([0.4, 0, 0.4], [0, 0, 0, 1])
    arm.start_stream()
    arm.start_stream()
    receipt = arm.send_ee_target([0.41, 0, 0.4], [0, 0, 0, 2])
    assert receipt.accepted and receipt.space == "ee"
    assert np.isclose(arm.get_state().ee_quaternion_xyzw[-1], 1)
    assert arm.hold().ee_position[0] == 0.41
    with pytest.raises(ValueError):
        arm.send_ee_target([np.nan, 0, 0], [0, 0, 0, 1])
    with pytest.raises(ValueError):
        arm.send_ee_target([0.4, 0, 0.4], [0, 0, 0, 0])
    with pytest.raises(ValueError):
        arm.send_ee_target([2, 0, 0.4], [0, 0, 0, 1])
    with pytest.raises(RuntimeError):
        arm.move_joints(np.zeros(7))
    arm.stop_stream(timeout_s=0.1)
    arm.stop_stream()
    assert arm.get_status().stream_space is None


def test_finite_moves_and_pose_conversion(arm):
    assert arm.move_joints(np.ones(7) * 0.1).status == "succeeded"
    assert arm.move_ee([0.4, 0, 0.5], [0, 0, 0, 1]).status == "succeeded"
    assert arm.move_to_start().final_state.joint_positions.shape == (7,)
    assert arm.pose_from_joints(np.zeros(7)).vector.shape == (6,)
    assert arm.cancel_motion().status == "idle"
    with pytest.raises(ValueError):
        arm.move_joints([0, 1])
    with pytest.raises(ValueError):
        arm.move_ee([0.4, 0, 0.5], [0, 0, 0, 1], speed_factor=0)


def test_trajectory_completion_callback_and_cancel(arm):
    points = [
        {"time_s": 0, "position": [0.4, 0, 0.4], "quaternion_xyzw": [0, 0, 0, 1]},
        {"time_s": 0.025, "position": [0.41, 0, 0.4], "quaternion_xyzw": [0, 0, 0, 1]},
    ]
    samples = []
    result = arm.execute_ee_trajectory(points, on_sample=samples.append)
    assert result.status == "succeeded" and samples
    points[-1]["time_s"] = 1
    results = []
    thread = threading.Thread(target=lambda: results.append(arm.execute_ee_trajectory(points)))
    thread.start()
    deadline = time.monotonic() + 1
    while not arm.get_status().motion_running and time.monotonic() < deadline:
        time.sleep(0.001)
    assert arm.cancel_motion(timeout_s=0.5).status == "cancelled"
    thread.join(1)
    assert results[0].status == "cancelled"
    with pytest.raises(ValueError):
        arm.execute_ee_trajectory(points[:1])


def test_gripper_interfaces_and_tactile(arm):
    arm.start_stream()
    assert arm.gripper_close().accepted
    assert arm.get_state().gripper.commanded_open_ratio == 0
    assert arm.gripper_open(wait=False).space == "gripper"
    arm.wait_gripper(timeout=1)
    arm.set_gripper(0.25)
    assert not arm.gripper_busy
    arm.stop_gripper()
    left, right = arm.get_tactile_images()
    assert left.shape == right.shape == (224, 224, 3)
    assert arm.get_status().stream_space == "ee"
    with pytest.raises(ValueError):
        arm.set_gripper(1.1)


def test_gripper_busy_failure_and_worker_ownership(arm):
    release = threading.Event()

    def blocked(*args, **kwargs):
        release.wait(1)
        return True

    arm._backend.gripper_command = blocked
    arm.gripper_open(wait=False)
    assert arm.gripper_busy
    with pytest.raises(RuntimeError):
        arm.gripper_close(wait=False)
    with pytest.raises(TimeoutError):
        arm.wait_gripper(timeout=0.001)
    release.set()
    arm.wait_gripper(timeout=1)
    arm._backend.gripper_command = lambda *args, **kwargs: False
    with pytest.raises(RuntimeError, match="failed"):
        arm.gripper_close()
    # Clear the consumed failure before fixture cleanup; separate tests verify propagation.
    arm._gripper_future = None


def test_stop_timeout_never_reports_stopped(arm):
    arm.start_stream()
    arm._backend.dq[:] = 1
    with pytest.raises(TimeoutError):
        arm.stop_stream(timeout_s=0.01)
    assert arm.get_status().stream_space == "ee"
    arm._backend.dq[:] = 0


def test_public_contract_has_no_untested_methods():
    tree = ast.parse(Path(__file__).read_text())
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    public = {
        name
        for name, value in RoboticArmControler.__dict__.items()
        if callable(value) and (not name.startswith("_") or name in {"__enter__", "__exit__"})
    }
    assert public <= calls
