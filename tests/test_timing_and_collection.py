import copy
import json
from types import SimpleNamespace

import pytest

from control.collection.pipeline import run_collection
from control.collection.vr import VRResampler
from control.config import load_config, validate
from control.recording_writer import FreshCameraPair
from control.util.timing import FixedRate
from replay.scripts.read_action_trace import read_action_trace
from replay.scripts.read_audio import read_audio
from replay.scripts.read_dataset import read_dataset


def test_fixed_rate_and_noninteger_camera_ratio():
    clock = [0]
    rate = FixedRate(
        100, clock=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0] + round(s * 1e9))
    )
    gate = FreshCameraPair()
    fresh = 0
    for _ in range(1000):
        ns = rate.tick()
        camera = ns * 30 // 1_000_000_000 + 1
        fresh += gate.accept(camera, camera)
    assert clock[0] == 9_990_000_000 and fresh == 300
    clock[0] += 1_000_000_000
    rate.tick()
    assert rate.overruns == 1


def test_vr_edges_and_stale_input():
    raw = SimpleNamespace(
        pos_x=0.0,
        pos_y=0.0,
        pos_z=0.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        quat_w=1.0,
        arm_enabled=True,
        pose_monotonic_ns=100,
        pose_seq=1,
        gripper_velocity_axis=0.0,
        y_pressed=True,
        x_pressed=False,
        gripper_open=False,
        gripper_close=False,
    )
    resampler = VRResampler(0.5)
    assert resampler.sample(raw, 101).save
    assert not resampler.sample(raw, 102).save
    assert not resampler.sample(raw, 1_000_000_000).enabled


@pytest.mark.parametrize(
    "space,tactile,audio",
    [("joint", False, True), ("ee", False, False), ("joint", True, False), ("ee", True, True)],
)
def test_complete_multirate_recording(tmp_path, space, tactile, audio):
    cfg = copy.deepcopy(load_config())
    cfg["control"]["frequency_hz"] = 100.0
    cfg["dataset"].update(root=str(tmp_path), date=space, action_space=space)
    cfg["gripper"]["type"] = "dh5" if tactile else "franka"
    cfg["tactile"]["enabled"] = tactile
    cfg["audio"].update(enabled=audio, vad_enabled=False)
    stats = run_collection(cfg, dry_run=True, max_steps=50)
    root = tmp_path / f"openpi/franka_lerobot_{space}"
    info = read_dataset(root)
    assert stats["ticks"] == 50
    assert info["fps"] == 30 and info["total_episodes"] == 1
    assert 10 <= info["total_frames"] <= 17
    assert next(f for f in info["features"] if f["key"] == "actions")["shape"] == [
        8 if space == "joint" else 7
    ]
    trace = read_action_trace(root, 0)
    assert trace["total_points"] >= 40
    if audio:
        assert read_audio(root, 0)["sample_rate"] == 16000
    assert ("gripper_image_left" in {f["key"] for f in info["features"]}) == tactile
    sync = next(root.rglob("*.sync.json"))
    payload = json.loads(sync.read_text())
    assert payload["control_mode"] == "ee" and payload["action_space"] == space
    assert all(
        r["action_sample_monotonic_ns"] > r["host_frame_monotonic_ns"]
        for r in payload["frame_records"]
    )
    # Changing labels/fps must never silently append to an existing dataset.
    cfg["dataset"]["action_space"] = "ee" if space == "joint" else "joint"
    with pytest.raises(ValueError, match="schema/fps"):
        run_collection(cfg, dry_run=True, max_steps=5)


def test_config_rejects_joint_streaming():
    cfg = load_config()
    cfg["control"]["mode"] = "joint"
    with pytest.raises(ValueError):
        validate(cfg)
