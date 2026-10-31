"""Episode phase boundaries and controller end/save interactions."""

import json
from dataclasses import replace

import numpy as np
import pytest

from control.collection.dataset import open_dataset
from control.collection.devices import CameraPair, FakeVR
from control.collection.pipeline import run_collection
from control.collection.recording import EpisodeRecorder
from control.collection.vr import EEMapper, VRResampler
from control.direct.fake import FakeBackend
from control.inference.loop import run_inference


def config_for(root):
    return {
        "dataset": {
            "repo_id": "local/interactions", "date": "test", "root": str(root),
            "instruction": "Insert the object", "action_space": "ee",
        },
        "robot": {"start_joint_position": [0, -0.5, 0, -2, 0, 1.5, 0.7]},
        "control": {"frequency_hz": 100},
        "camera": {"image_hw": 32, "fps": 30},
        "gripper": {"type": "none"},
        "audio": {"enabled": False},
        "inference": {
            "host": "127.0.0.1", "port": 8001, "action_space": "ee",
            "action_fps": 30, "model_horizon": 50, "execute_steps": 2,
            "enable_vr_takeover": True,
        },
    }


def test_phase_latches_once_resets_and_preserves_sample_boundary(tmp_path, caplog):
    config = config_for(tmp_path)
    dataset, root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, root, config)
    state = FakeBackend().snapshot()
    sample = VRResampler().sample(FakeVR().latest, state.sampled_monotonic_ns)
    image = np.ones((32, 32, 3), dtype=np.uint8)
    caplog.set_level("INFO")
    try:
        assert not recorder.enter_critical_phase(1)
        recorder.start()
        boundary = recorder.started_ns + 100
        recorder.trace(
            state, sample, state.ee_position, state.ee_quaternion_xyzw,
            command_submitted_ns=boundary - 10, trace_sample_ns=boundary - 1,
        )
        assert recorder.enter_critical_phase(boundary)
        assert not recorder.enter_critical_phase(boundary + 100)
        for sample_ns in (boundary - 1, boundary, boundary + 1):
            recorder.trace(
                state, sample, state.ee_position, state.ee_quaternion_xyzw,
                command_submitted_ns=boundary - 10, trace_sample_ns=sample_ns,
            )
        ns = state.sampled_monotonic_ns
        recorder.frame(state, CameraPair(image, image, ns, ns, 0, 0))
        recorder.finish(state)
        rows = [json.loads(line) for line in (root / "episode_000000.actions.jsonl").read_text().splitlines()]
        assert [row["in_critical_phase"] for row in rows] == [False, False, True, True]
        sync = json.loads((root / "episode_000000.sync.json").read_text())
        assert sync["critical_phase_annotation_version"] == 1
        assert sync["critical_phase_start_monotonic_ns"] == boundary
        assert caplog.text.count("已进入关键阶段") == 1
        assert not recorder.enter_critical_phase(boundary + 200)
        recorder.start()
        assert recorder.critical_phase_start_ns is None
        assert recorder.control_events == []
        recorder.trace(state, sample, state.ee_position, state.ee_quaternion_xyzw)
        ns += 1
        recorder.frame(state, CameraPair(image, image, ns, ns, 0, 0))
        recorder.finish(state)
        second = json.loads((root / "episode_000001.sync.json").read_text())
        assert second["critical_phase_start_monotonic_ns"] is None
        assert not json.loads((root / "episode_000001.actions.jsonl").read_text())["in_critical_phase"]
    finally:
        recorder.close()


class LocalPolicy:
    server_metadata = {
        "action_space": "ee", "action_fps": 30, "action_horizon": 50, "action_dim": 7,
    }

    def infer(self, observation):
        return {
            "actions": np.tile(np.r_[observation["observation.ee_pose"], 1.0], (50, 1)),
            "source": "RL", "actor_version": "v1",
        }

    def close(self):
        pass


@pytest.mark.parametrize("mark_after_start", [False, True])
def test_collection_ignores_idle_phase_press(tmp_path, monkeypatch, mark_after_start):
    original = VRResampler.sample
    tick = 0

    def sample(resampler, raw, now):
        nonlocal tick
        tick += 1
        return replace(
            original(resampler, raw, now), enabled=True,
            enter_critical_phase=tick == 1 or (mark_after_start and tick == 4),
        )

    def map_pose(mapper, sample, state, now):
        return (
            state.ee_position + ([0.001, 0, 0] if tick >= 2 else [0, 0, 0]),
            state.ee_quaternion_xyzw,
        )

    monkeypatch.setattr(VRResampler, "sample", sample)
    monkeypatch.setattr(EEMapper, "map", map_pose)
    run_collection(config_for(tmp_path), dry_run=True, max_steps=6)
    sync = json.loads(next(tmp_path.rglob("*.sync.json")).read_text())
    rows = [json.loads(line) for line in next(tmp_path.rglob("*.actions.jsonl")).read_text().splitlines()]
    assert len(rows) == 5  # Motion starts at tick 2; tick 1 must not start recording.
    assert (sync["critical_phase_start_monotonic_ns"] is not None) is mark_after_start
    assert [row["in_critical_phase"] for row in rows] == (
        [False, False, True, True, True] if mark_after_start else [False] * 5
    )


@pytest.mark.parametrize("discard", [False, True])
def test_collection_save_and_discard_both_return_to_start_before_rearming(
    tmp_path, monkeypatch, discard
):
    from control.collection import pipeline
    from control.collection.vr import ResampledVR

    config = config_for(tmp_path)
    config["robot"].update(backend="ros2", move_to_start=True)
    backend = FakeBackend()
    monkeypatch.setattr("control.direct.fake.FakeBackend", lambda: backend)
    moves = []
    original_move = backend.move_joints
    original_start = EpisodeRecorder.start
    starts = []
    boundaries = []
    finish = pipeline.AsyncEpisodeRecorder.finish

    def finish_episode(recorder, state, **kwargs):
        finish(recorder, state, **kwargs)
        boundaries.append(not recorder.active)

    def move(target, speed):
        # The capture boundary ends immediately; disk publication may still run.
        if moves:
            assert boundaries[-1]
        moves.append(np.array(target))
        return original_move(target, speed)

    def start(recorder, **kwargs):
        starts.append(True)
        original_start(recorder, **kwargs)

    tick = 0

    def sample(resampler, raw, now):
        nonlocal tick
        tick += 1
        return ResampledVR(
            position=np.array([tick * 0.01, 0, 0]), quaternion_xyzw=np.array([0, 0, 0, 1]),
            enabled=tick != 6, save=tick == 4 and not discard, discard=tick == 4 and discard,
            close=False, open=False, axis=0, source_ns=now, source_seq=tick, released=tick == 6,
        )

    monkeypatch.setattr(pipeline.EpisodeRecorder, "start", start)
    monkeypatch.setattr(pipeline.AsyncEpisodeRecorder, "finish", finish_episode)
    monkeypatch.setattr(backend, "move_joints", move)
    monkeypatch.setattr(VRResampler, "sample", sample)
    run_collection(config, dry_run=True, max_steps=6)
    assert len(moves) == 2  # Startup and either save or discard.
    for target in moves:
        np.testing.assert_array_equal(target, config["robot"]["start_joint_position"])
    assert len(starts) == 1  # Holding the deadman across reset cannot start another episode.
    assert bool(list(tmp_path.rglob("*.sync.json"))) is not discard


@pytest.mark.parametrize("discard", [False, True])
def test_inference_controller_end_and_phase_with_same_episode_takeover(tmp_path, monkeypatch, discard):
    original = VRResampler.sample
    tick = 0

    def sample(resampler, raw, now):
        nonlocal tick
        tick += 1
        return replace(
            original(resampler, raw, now), enabled=5 <= tick <= 8,
            enter_critical_phase=tick in (3, 4),
            save=tick == 26 and not discard, discard=tick == 26 and discard,
        )

    backend = FakeBackend()
    monkeypatch.setattr(VRResampler, "sample", sample)
    monkeypatch.setattr("control.direct.fake.FakeBackend", lambda: backend)
    result = run_inference(config_for(tmp_path), dry_run=True, max_steps=60, policy=LocalPolicy())
    assert result["ticks"] == 25
    assert backend.calls[-2:] == ["move_joints", "close"]
    if discard:
        assert "episode" not in result
        assert not list(tmp_path.rglob("*.sync.json"))
        assert not list(tmp_path.rglob("*.actions.jsonl"))
        return
    root = tmp_path / result["episode"]["dataset_id"]
    sync = json.loads((root / "episode_000000.sync.json").read_text())
    assert sync["success"] is None
    assert [event["type"] for event in sync["control_events"]] == ["takeover_start", "takeover_end"]
    assert sync["control_events"][0]["host_monotonic_ns"] < sync["control_events"][1]["host_monotonic_ns"]
    rows = [json.loads(line) for line in (root / "episode_000000.actions.jsonl").read_text().splitlines()]
    assert len(rows) == 25
    assert not any(row["in_critical_phase"] for row in rows[:2])
    assert all(row["in_critical_phase"] for row in rows[2:])
    assert all(row["control_source"] == "HUMAN" for row in rows[4:8])
    assert "RL" in [row["control_source"] for row in rows[8:]]
