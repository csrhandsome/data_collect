from fastapi.testclient import TestClient

from control.collection.controller_hints import build_controller_hints
from control.config import load_config
from control.vr_input import VRInputReader


def actions(hints, hand):
    return {row["button"]: row["action"] for row in hints[hand]["rows"]}


def test_collection_instructions_match_save_discard_and_franka_controls():
    config = load_config()
    config["gripper"]["type"] = "franka"
    hints = build_controller_hints(config)
    left, right = actions(hints, "left"), actions(hints, "right")
    assert left["Y"] == "保存当前片段并回位"
    assert left["X"] == "丢弃当前片段并回位"
    assert "0.5 秒" in left["双食指扳机"]
    assert right["A"] == "闭合夹爪"
    assert right["B"] == "张开夹爪"
    assert not any("摇杆" in button for button in right)
    reader = VRInputReader(controller_hints=hints)
    with TestClient(reader._teleop.app) as client:
        with client.websocket_connect("/ws") as ws:
            message = ws.receive_json()
            assert message["data"]["controller_hints"] == hints


def test_dh5_hints_include_only_supported_step_controls():
    hints = build_controller_hints({"gripper": {"type": "dh5"}, "vr": {"long_press_s": 1}})
    assert actions(hints, "right")["摇杆 ↑"] == "逐步闭合夹爪"
    assert actions(hints, "right")["摇杆 ↓"] == "逐步张开夹爪"
    assert "1 秒" in actions(hints, "left")["双食指扳机"]


def test_disabled_reset_hints_describe_holding_position():
    hints = build_controller_hints({"robot": {"move_to_start": False}})
    left = actions(hints, "left")
    assert left["Y"] == "保存当前片段，保持当前位置"
    assert left["X"] == "丢弃当前片段，保持当前位置"


def test_disabled_recording_or_gripper_does_not_advertise_inactive_buttons():
    hints = build_controller_hints(
        {"dataset": {"enable_logging": False}, "gripper": {"type": "none"}}
    )
    assert not {"X", "Y"} & actions(hints, "left").keys()
    assert not {"A", "B"} & actions(hints, "right").keys()
    assert build_controller_hints({"vr": {"show_controller_hints": False}}) == {}


def test_takeover_hints_do_not_claim_save_buttons_or_pause_policy():
    hints = build_controller_hints({"inference": {"enable_vr_takeover": True}})
    assert not {"X", "Y"} & actions(hints, "left").keys()
    assert actions(hints, "left")["松开任一扳机"] == "交回策略控制"
    assert actions(hints, "right")["夹爪操作"] == "仅人工接管时生效"
