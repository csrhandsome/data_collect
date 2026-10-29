import json
from dataclasses import replace

import numpy as np
import pytest

from control.collection.audio import SyntheticMicrophone
from control.collection.dataset import open_dataset
from control.collection.deletion import EpisodeDeleter
from control.collection.devices import CameraPair, FakeVR
from control.collection.recording import EpisodeRecorder
from control.collection.vr import VRResampler
from control.direct.fake import FakeBackend
from data_analysis.dataset_io import episode_dataframe, episode_paths
from replay.scripts.read_dataset import read_dataset


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_ros_wrench_is_saved_in_parquet_and_traces(tmp_path, enabled):
    config = {
        "robot": {"backend": "ros2"},
        "dataset": {"repo_id": "local/wrench", "date": "capture", "root": str(tmp_path),
                    "instruction": "Contact", "action_space": "ee"},
        "camera": {"image_hw": 32, "fps": 30},
        "control": {"frequency_hz": 100},
        "observation": {
            "exterior_image": False, "wrist_image_left": False,
            "joint_position": False, "ee_pose": False, "ee_position": True,
            "gripper_position": False, "ee_wrench": enabled,
        },
        "action": {"joint_position": False, "ee_pose": False, "gripper_position": False},
    }
    dataset, root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, root, config)
    base = FakeBackend().snapshot()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    expected = []
    try:
        recorder.start()
        for index in range(3):
            wrench = np.array([index + 1, -2, 3, 0.1, -0.2, 0.3])
            expected.append(wrench)
            state = replace(base, ee_wrench=wrench, sampled_monotonic_ns=base.sampled_monotonic_ns + index)
            sample = VRResampler().sample(FakeVR().latest, state.sampled_monotonic_ns)
            recorder.trace(state, sample, state.ee_position, state.ee_quaternion_xyzw)
            ns = state.sampled_monotonic_ns
            recorder.frame(state, CameraPair(image, image, ns, ns, 0, 0))
        recorder.finish(state)
        recorder.finalize()
    finally:
        recorder.close()
    rows = episode_dataframe(episode_paths(root)[0], 0)
    sync = json.loads((root / "episode_000000.sync.json").read_text())
    trace = [json.loads(line) for line in (root / sync["action_trace"]).read_text().splitlines()]
    assert ("observation.ee_wrench" in rows) is enabled
    if enabled:
        np.testing.assert_allclose(np.stack(rows["observation.ee_wrench"]), expected, rtol=1e-6)
        assert np.stack(rows["observation.ee_wrench"]).dtype == np.float32
        np.testing.assert_allclose([row["ee_wrench"] for row in trace], expected)
        np.testing.assert_allclose([row["ee_wrench"] for row in sync["frame_records"]], expected)
    else:
        assert all("ee_wrench" not in row for row in trace + sync["frame_records"])


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("discard_between", [False, True])
def test_same_recorder_saves_consecutive_episodes(tmp_path, audio, discard_between):
    config = {
        "dataset": {
            "repo_id": "local/recording",
            "date": "consecutive",
            "root": str(tmp_path),
            "instruction": "Move the object",
            "action_space": "ee",
        },
        "camera": {"image_hw": 32, "fps": 30},
        "control": {"frequency_hz": 100},
    }
    dataset, root = open_dataset(config)
    microphone = SyntheticMicrophone({}) if audio else None
    recorder = EpisodeRecorder(dataset, root, config, microphone)
    writer = recorder.writer
    backend = FakeBackend()
    vr = FakeVR()
    resampler = VRResampler()

    def record_episode(value, save=True):
        backend.position[0] = value
        recorder.start()
        for index in range(3):
            state = backend.snapshot()
            sample = resampler.sample(vr.latest, state.sampled_monotonic_ns)
            recorder.trace(state, sample, state.ee_position, state.ee_quaternion_xyzw)
            image = np.full((32, 32, 3), index + 1, dtype=np.uint8)
            ns = state.sampled_monotonic_ns
            recorder.frame(state, CameraPair(image, image.copy(), ns, ns, ns / 1e6, ns / 1e6))
        recorder.finish(backend.snapshot(), save=save, success=save)

    try:
        record_episode(0.4)
        assert recorder.dataset.meta.total_episodes == 1
        assert recorder.dataset.meta.total_frames == 3
        if discard_between:
            record_episode(0.6, save=False)
            assert recorder.dataset.meta.total_episodes == 1
            assert not (root / "episode_000001.actions.jsonl").exists()
        record_episode(0.5)
        assert recorder.dataset is dataset and recorder.writer is writer
        recorder.finalize()
        info = read_dataset(root)
        assert info["total_episodes"] == 2
        assert info["total_frames"] == 6
        for index, path in episode_paths(root).items():
            rows = episode_dataframe(path, index, columns=["observation.ee_position", "frame_index"])
            assert rows["frame_index"].tolist() == [0, 1, 2]
            np.testing.assert_allclose(np.stack(rows["observation.ee_position"])[:, 0], 0.4 + index * 0.1)
            sync_dir = root / "audio" if audio else root
            sync = json.loads((sync_dir / f"episode_{index:06d}.sync.json").read_text())
            assert sync["video_frames"] == 3
            assert sync["action_records"] == 3
            assert sync["frame_records"][-1]["terminal_action"]
            if audio:
                assert (root / sync["audio_path"]).is_file()
        assert not list(root.rglob("*.sync.json.tmp"))
        assert len(set(episode_paths(root).values())) == 1

        # Editing a sealed session must be reflected when recording resumes.
        recorder.finalize()
        EpisodeDeleter(root).delete_episode(0)
        record_episode(0.7)
        assert recorder.dataset is not dataset
        recorder.close()
        recorder.close()
        info = read_dataset(root)
        assert info["total_episodes"] == 2 and info["total_frames"] == 6
        for index, expected in enumerate((0.5, 0.7)):
            rows = episode_dataframe(episode_paths(root)[index], index, columns=["observation.ee_position"])
            np.testing.assert_allclose(np.stack(rows["observation.ee_position"])[:, 0], expected)
        with pytest.raises(RuntimeError, match="closed"):
            recorder.start()
    finally:
        recorder.close()


def test_finalize_requires_finishing_active_episode_and_close_discards_it(tmp_path):
    config = {
        "dataset": {"repo_id": "local/recording", "date": "active", "root": str(tmp_path),
                    "instruction": "Move", "action_space": "ee"},
        "camera": {"image_hw": 32, "fps": 30},
        "control": {"frequency_hz": 100},
    }
    dataset, root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, root, config)
    try:
        recorder.start()
        with pytest.raises(RuntimeError, match="active episode"):
            recorder.finalize()
        state = FakeBackend().snapshot()
        image = np.ones((32, 32, 3), dtype=np.uint8)
        for offset in (0, 1):
            ns = state.sampled_monotonic_ns + offset
            recorder.frame(state, CameraPair(image, image, ns, ns, ns / 1e6, ns / 1e6))
    finally:
        recorder.close()
    assert dataset.meta.total_episodes == 0
    assert not list((root / "images").rglob("*.png"))


def test_trace_uses_accepted_gripper_target_and_actual_controller(tmp_path):
    config = {
        "dataset": {"repo_id": "local/recording", "date": "trace", "root": str(tmp_path),
                    "instruction": "Move", "action_space": "ee"},
        "camera": {"image_hw": 32, "fps": 30},
        "control": {"frequency_hz": 100},
    }
    dataset, root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, root, config)
    state = FakeBackend().snapshot()
    sample = VRResampler().sample(FakeVR().latest, state.sampled_monotonic_ns)
    try:
        recorder.start()
        for source in ("RL", "HUMAN", "RL"):
            recorder.trace(
                state, sample, state.ee_position, state.ee_quaternion_xyzw,
                control_source=source, target_gripper_position=0.2,
                actor_version="v2" if source == "RL" else None,
            )
        recorder.trace_writer.drain()
        recorder.trace_sink.file.flush()
        rows = [json.loads(line) for line in recorder.trace_path.read_text().splitlines()]
        assert [row["control_source"] for row in rows] == ["RL", "HUMAN", "RL"]
        assert all(row["target_gripper_position"] == 0.2 for row in rows)
        assert all(row["gripper_position"] == 1 for row in rows)
        assert [row["intervention_flag"] for row in rows] == [False, True, False]
    finally:
        recorder.close()
