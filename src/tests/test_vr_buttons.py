import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from control import vr_input
from control.collection.vr import VRResampler
from control.ros2.vr_connector import dumps_vr, loads_vr
from control.vr_input import VRInput, VRInputProcess, VRInputReader


def xr_message(left_pressed=(), right_pressed=()):
    def controller(hand, pressed):
        return {
            "role": "controller",
            "handedness": hand,
            "gamepad": {
                "buttons": [
                    {"pressed": index in pressed, "touched": False, "value": float(index in pressed)}
                    for index in range(6)
                ],
                "axes": [0.0, 0.0, 0.2, -0.4] if hand == "right" else [],
            },
        }

    return {
        "data": {
            "timestamp_unix_ms": 0,
            "devices": [controller("left", left_pressed), controller("right", right_pressed)],
        }
    }


def test_left_thumbstick_click_parsing_preserves_existing_controls():
    reader = VRInputReader(long_press_s=0.0)
    reader._on_xr_update(None, xr_message(left_pressed=(0, 3, 5), right_pressed=(0, 4)))
    sample = reader.latest
    assert sample.critical_pressed
    assert sample.arm_enabled
    assert sample.y_pressed and sample.save_pressed and not sample.x_pressed
    assert sample.gripper_close and not sample.gripper_open
    assert sample.gripper_velocity_axis == pytest.approx(0.4)

    reader._on_xr_update(None, xr_message(left_pressed=(4,), right_pressed=(3, 5)))
    sample = reader.latest
    assert not sample.critical_pressed  # Right stick click has no phase meaning.
    assert sample.x_pressed and not sample.y_pressed and not sample.save_pressed
    assert sample.gripper_open and not sample.gripper_close
    assert not sample.arm_enabled


def test_missing_controller_buttons_do_not_enter_critical_phase():
    reader = VRInputReader()
    reader._on_xr_update(None, {"data": {"timestamp_unix_ms": 0, "devices": []}})
    assert not reader.latest.critical_pressed


def test_critical_click_is_one_fresh_edge_and_stale_held_input_cannot_retrigger():
    resampler = VRResampler(stale_after_s=0.5)
    raw = VRInput(critical_pressed=True, pose_monotonic_ns=1_000_000_000, pose_seq=1)
    assert resampler.sample(raw, 1_000_000_000).enter_critical_phase
    assert not resampler.sample(raw, 1_010_000_000).enter_critical_phase
    assert not resampler.sample(raw, 1_500_000_000).enter_critical_phase

    raw.pose_monotonic_ns = 1_600_000_000
    raw.pose_seq += 1
    assert not resampler.sample(raw, 1_600_000_000).enter_critical_phase
    raw.critical_pressed = False
    assert not resampler.sample(raw, 1_610_000_000).enter_critical_phase
    raw.critical_pressed = True
    assert resampler.sample(raw, 1_620_000_000).enter_critical_phase


def test_initial_stale_click_is_not_queued_for_reconnection():
    resampler = VRResampler(stale_after_s=0.5)
    raw = VRInput(critical_pressed=True, pose_monotonic_ns=1_000_000_000)
    assert not resampler.sample(raw, 1_500_000_000).enter_critical_phase
    raw.pose_monotonic_ns = 1_600_000_000
    assert not resampler.sample(raw, 1_600_000_000).enter_critical_phase


def test_critical_click_roundtrips_through_worker_shared_memory(monkeypatch):
    reader = VRInputReader()
    reader._on_xr_update(None, xr_message(left_pressed=(3, 5), right_pressed=(4,)))
    process = VRInputProcess()
    monkeypatch.setattr(vr_input, "VRInputReader", lambda **kwargs: reader)
    monkeypatch.setattr(vr_input.threading, "Thread", lambda **kwargs: SimpleNamespace(start=Mock()))

    class OneSnapshotComplete(Exception):
        pass

    def stop_after_snapshot(_seconds):
        raise OneSnapshotComplete

    monkeypatch.setattr(vr_input.time, "sleep", stop_after_snapshot)
    with pytest.raises(OneSnapshotComplete):
        process._vr_worker(process._shm, "localhost", 4443, 0.5, 1.0, 1.0, {})
    sample = process.latest
    assert sample.critical_pressed
    assert sample.y_pressed and sample.gripper_close
    assert sample.raw_quaternion_xyzw == reader.latest.raw_quaternion_xyzw
    assert sample.pose_seq == reader.latest.pose_seq


def test_critical_click_roundtrips_through_ros2_payload_and_resampling():
    reader = VRInputReader()
    reader._on_xr_update(None, xr_message(left_pressed=(3,)))
    sample = loads_vr(dumps_vr(reader.latest))
    assert sample.critical_pressed
    assert VRResampler().sample(sample, sample.pose_monotonic_ns).enter_critical_phase

    old_payload = json.loads(dumps_vr(reader.latest))
    del old_payload["critical_pressed"]
    sample = loads_vr(json.dumps(old_payload))
    assert not sample.critical_pressed
    assert not VRResampler().sample(sample, sample.pose_monotonic_ns).enter_critical_phase
