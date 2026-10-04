import json

import numpy as np
import pytest

from control._panda.fake import FakeBackend
from control.collection.audio import SyntheticMicrophone
from control.collection.dataset import open_dataset
from control.collection.deletion import EpisodeDeleter
from control.collection.devices import CameraPair, FakeVR
from control.collection.recording import EpisodeRecorder
from control.collection.vr import VRResampler
from data_analysis.dataset_io import episode_dataframe, episode_paths
from replay.scripts.read_dataset import read_dataset


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
            rows = episode_dataframe(path, index, columns=["ee_position", "frame_index"])
            assert rows["frame_index"].tolist() == [0, 1, 2]
            np.testing.assert_allclose(np.stack(rows["ee_position"])[:, 0], 0.4 + index * 0.1)
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
            rows = episode_dataframe(episode_paths(root)[index], index, columns=["ee_position"])
            np.testing.assert_allclose(np.stack(rows["ee_position"])[:, 0], expected)
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
