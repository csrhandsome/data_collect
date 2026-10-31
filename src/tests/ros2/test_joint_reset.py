"""Joint reset must follow measured feedback and stop before VR can resume."""

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from control.direct.fake import FakeBackend
from control.ros2 import joint_motion
from control.util.robot import reset_robot

START = np.array([0, -0.5, 0, -2, 0, 1.5, 0.7])


class ResetPeer:
    def __init__(self, monkeypatch, *, follows=True, cancel_at=None, stale_at=None):
        self.now = 0.0
        self.q = START.copy()
        self.commands = []
        self.holds = []
        self.follows = follows
        self.cancel_at = cancel_at
        self.stale_at = stale_at
        self._streaming = False
        self.config = {"robot": {"arrival_timeout_s": 0.3}}
        self.cfg = {}
        self._source = SimpleNamespace(wait=lambda predicate, timeout: self.wait_for_peer(predicate))
        self.subscribers = 1
        self._joint_publisher = SimpleNamespace(get_subscription_count=lambda: self.subscribers)
        self._joint_cancel = SimpleNamespace(
            clear=lambda: None,
            is_set=lambda: self.cancel_at is not None and self.now >= self.cancel_at,
            wait=self.wait,
        )
        monkeypatch.setattr(joint_motion, "time", SimpleNamespace(monotonic=lambda: self.now))

    def wait_for_peer(self, predicate):
        if not predicate():
            raise TimeoutError("No subscriber")

    def wait(self, duration):
        self.now += duration

    def snapshot(self):
        if self.stale_at is not None and self.now >= self.stale_at:
            raise TimeoutError("Franka feedback is stale")
        return replace(FakeBackend().snapshot(), joint_positions=self.q)

    def publish_joints(self, desired):
        self.commands.append((self.now, desired.copy()))
        if self.follows:
            self.q = desired.copy()

    def target(self, position, quaternion, nullspace):
        self.holds.append((position.copy(), quaternion.copy()))


def test_joint_reset_is_bounded_waits_for_arrival_and_restores_cartesian_hold(monkeypatch):
    peer = ResetPeer(monkeypatch)
    target = START + 0.1
    assert joint_motion.move_joints(peer, target, 0.2)
    np.testing.assert_allclose(peer.q, target)
    times, positions = zip(*peer.commands)
    velocity = np.diff(np.array(positions), axis=0) / np.diff(times)[:, None]
    assert np.abs(velocity).max() <= 0.2
    assert peer.now >= 2.2  # Duration plus stable measured arrival.
    assert len(peer.holds) == 1
    assert not peer._streaming


@pytest.mark.parametrize("distance,error", [(0.4, ValueError), (0.1, TimeoutError)])
def test_joint_reset_never_reports_arrival_for_nonfollowing_feedback(monkeypatch, distance, error):
    peer = ResetPeer(monkeypatch, follows=False)
    with pytest.raises(error) as failure:
        joint_motion.move_joints(peer, START + distance, 0.2)
    if error is TimeoutError:
        for field in ("max_position_error=", "max_velocity=", "target_q=", "error_q=", "measured_dq="):
            assert field in str(failure.value)
    assert len(peer.holds) == 1
    assert not peer._streaming


def test_cancelled_reset_stops_publishing_and_holds(monkeypatch):
    peer = ResetPeer(monkeypatch, cancel_at=0.1)
    assert not joint_motion.move_joints(peer, START + 0.1, 0.2)
    assert peer.commands[-1][0] < 0.1
    assert len(peer.holds) == 1
    assert not peer._streaming


def test_stale_feedback_aborts_reset_without_more_joint_targets(monkeypatch):
    peer = ResetPeer(monkeypatch, stale_at=0.1)
    with pytest.raises(TimeoutError, match="stale"):
        joint_motion.move_joints(peer, START + 0.1, 0.2)
    assert peer.commands[-1][0] < 0.1
    assert not peer._streaming


@pytest.mark.parametrize("target", [[0] * 7, [float("nan")] * 7, [0] * 6])
def test_bad_joint_target_never_reaches_controller(monkeypatch, target):
    peer = ResetPeer(monkeypatch)
    with pytest.raises(ValueError):
        joint_motion.move_joints(peer, target, 0.2)
    assert peer.commands == peer.holds == []


def test_old_controller_without_joint_reset_is_rejected_before_motion(monkeypatch):
    peer = ResetPeer(monkeypatch)
    peer.subscribers = 0
    with pytest.raises(RuntimeError, match="rebuild"):
        joint_motion.move_joints(peer, START, 0.2)
    assert peer.commands == peer.holds == []


def test_cancelled_workflow_reset_does_not_restart_vr_stream():
    events = []
    arm = SimpleNamespace(
        _config={"robot": {"move_to_start": True}},
        wait_gripper=lambda **kwargs: events.append("wait_gripper"),
        stop_stream=lambda: events.append("stop_stream"),
        get_capabilities=lambda: SimpleNamespace(gripper=False),
        move_to_start=lambda: SimpleNamespace(status="cancelled"),
        start_stream=lambda: events.append("start_stream"),
    )
    with pytest.raises(RuntimeError, match="not completed"):
        reset_robot(arm)
    assert events == ["wait_gripper", "stop_stream"]


def test_preflight_rejects_controller_with_old_joint_damping(tmp_path):
    source = Path(__file__).resolve().parents[2] / "env/check_controller.py"
    spec = importlib.util.spec_from_file_location("controller_preflight", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    library = tmp_path / "controller.so"
    library.write_bytes(b"filter_coefficient ~/joint_reset")
    with pytest.raises(RuntimeError, match="paired joint damping"):
        module.check_installed(library)
    library.write_bytes(library.read_bytes() + b" Joint reset paired damping:")
    module.check_installed(library)
