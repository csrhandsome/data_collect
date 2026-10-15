"""Exercise upstream v3 writers and project tooling on real temporary files."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from control.collection.dataset import open_dataset
from control.collection.deletion import EpisodeDeleter
from control.collection.pipeline import run_collection
from control.config import load_config
from data_analysis.dataset_io import episode_dataframe, episode_paths, episode_rows
from data_analysis.merge_lerobot_datasets import merge_datasets
from replay.scripts.read_audio import read_audio
from replay.scripts.read_dataset import read_dataset


@pytest.fixture
def v3_collection(tmp_path):
    config = load_config()
    config["dataset"].update(root=str(tmp_path), date="upgrade")
    config["camera"]["image_hw"] = 32
    config["audio"].update(enabled=True, vad_enabled=False)
    for index in range(3):
        config["dataset"]["instruction"] = f"Task {index}"
        run_collection(config, dry_run=True, max_steps=20)
    return config, tmp_path / "franka_lerobot_upgrade"


def test_v3_deletion_rewrites_shared_data_and_sidecars(v3_collection):
    config, root = v3_collection
    before = read_dataset(root)
    result = EpisodeDeleter(root).delete_episode(1)
    after = read_dataset(root)
    assert after["version"] == "v3.0" and after["total_episodes"] == 2
    assert after["total_frames"] == before["total_frames"] - result.deleted_frames
    assert after["episodes"][1]["tasks"] == ["Task 2"]
    assert read_audio(root, 1)["sample_rate"] == 16000
    sync = json.loads((root / "audio/episode_000001.sync.json").read_text())
    assert sync["episode_index"] == 1 and sync["action_trace"] == "episode_000001.actions.jsonl"
    dataset, _ = open_dataset(config)
    dataset.finalize()
    EpisodeDeleter(root).delete_episode(1)
    EpisodeDeleter(root).delete_episode(0)
    assert read_dataset(root)["total_episodes"] == 0
    dataset, _ = open_dataset(config)
    dataset.finalize()


def _snapshot(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def test_v3_deletion_copies_only_auxiliary_files_and_preserves_annotations(v3_collection, monkeypatch):
    _, root = v3_collection
    (root / "notes.txt").write_text("Keep this dataset note")
    for index in range(3):
        clips = root / f"audio/vad_segments/episode_{index:06d}"
        clips.mkdir(parents=True)
        (clips / "seg_000.wav").write_bytes(bytes([index]))
        sync_path = root / f"audio/episode_{index:06d}.sync.json"
        sync = json.loads(sync_path.read_text())
        sync["vad_metadata"] = {"segments_dir": str(clips)}
        sync["vad_segments"] = [{
            "seg_id": f"episode_{index:06d}_vad_000",
            "audio_wav_path": str(clips / "seg_000.wav"),
            "asr_transcript": f"episode_{index:06d} stays in the text",
            "verified": True,
        }]
        sync_path.write_text(json.dumps(sync))
    copied = []
    copyfile = shutil.copyfile

    def copy(source, destination, *args, **kwargs):
        copied.append(str(source))
        return copyfile(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", copy)
    EpisodeDeleter(root).delete_episode(1)
    assert copied
    assert not any(Path(path).is_relative_to(root / "data") for path in copied)
    assert not any(Path(path).is_relative_to(root / "meta") for path in copied)
    assert not any("episode_000001" in path for path in copied)
    assert (root / "notes.txt").read_text() == "Keep this dataset note"
    assert (root / "audio/vad_segments/episode_000001/seg_000.wav").read_bytes() == b"\x02"
    sync = json.loads((root / "audio/episode_000001.sync.json").read_text())
    segment = sync["vad_segments"][0]
    assert segment["seg_id"] == "episode_000001_vad_000"
    assert segment["asr_transcript"] == "episode_000002 stays in the text"
    assert segment["verified"] is True
    assert Path(segment["audio_wav_path"]).is_file()
    assert Path(sync["vad_metadata"]["segments_dir"]).is_dir()
    assert not list(root.parent.glob(f".{root.name}.delete-*"))


@pytest.mark.parametrize("failure", ["generate", "sidecar", "patch", "validate", "fsync", "commit", "interrupt"])
def test_v3_deletion_failure_keeps_source_byte_identical(v3_collection, monkeypatch, failure):
    from lerobot.datasets import dataset_tools

    from control.collection import _episode_deletion as deletion

    _, root = v3_collection
    before = _snapshot(root)

    def fail(*args, **kwargs):
        raise OSError("injected v3 deletion failure")

    if failure in {"generate", "interrupt"}:
        generate = dataset_tools.delete_episodes

        def fail_after_generating(*args, **kwargs):
            generate(*args, **kwargs)
            if failure == "interrupt":
                raise KeyboardInterrupt
            fail()

        monkeypatch.setattr(dataset_tools, "delete_episodes", fail_after_generating)
    elif failure == "sidecar":
        monkeypatch.setattr(shutil, "copy2", fail)
    elif failure == "patch":
        monkeypatch.setattr(deletion, "_patch_sync_json", fail)
    elif failure == "validate":
        monkeypatch.setattr(deletion, "_validate_result", fail)
    elif failure == "fsync":
        monkeypatch.setattr(deletion.os, "fsync", fail)
    else:
        monkeypatch.setattr(deletion, "_exchange_directories", fail)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else OSError):
        EpisodeDeleter(root).delete_episode(1)
    assert _snapshot(root) == before
    assert not list(root.parent.glob(f".{root.name}.delete-*"))


def test_v3_concurrent_source_change_cancels_commit(v3_collection, monkeypatch):
    from control.collection import _episode_deletion as deletion

    _, root = v3_collection
    before = _snapshot(root)
    validate = deletion._validate_result

    def change_source(*args):
        validate(*args)
        (root / "new-record.txt").write_text("new data")

    monkeypatch.setattr(deletion, "_validate_result", change_source)
    with pytest.raises(RuntimeError, match="其他进程修改"):
        EpisodeDeleter(root).delete_episode(1)
    after = _snapshot(root)
    assert after.pop("new-record.txt")
    assert after == before


def test_v3_task_rewrite_updates_every_episode_in_shared_shard(v3_collection):
    from lerobot.datasets import LeRobotDataset

    from data_analysis.dataset_io import rewrite_v3_tasks

    _, root = v3_collection
    prompts = {0: "Command A", 1: "Command B", 2: "Command A"}
    result = rewrite_v3_tasks(root, prompts)
    assert result["num_tasks"] == 2
    paths = episode_paths(root)
    for row in episode_rows(root):
        index = row["episode_index"]
        assert row["tasks"] == [prompts[index]]
        assert set(episode_dataframe(paths[index], index)["task_index"]) == {int(index == 1)}
    dataset = LeRobotDataset(root.name, root=root, video_backend="pyav")
    assert dataset.meta.get_task_index("Command B") == 1


def test_v3_merge_retains_audio_and_trace(v3_collection, tmp_path):
    _, root = v3_collection
    output = merge_datasets(
        [str(root), str(root)], output_name="merged", data_root=tmp_path, overwrite=False
    )
    info = read_dataset(output)
    assert info["total_episodes"] == 6
    assert info["total_frames"] == 2 * read_dataset(root)["total_frames"]
    for index in range(6):
        assert read_audio(output, index)["sample_rate"] == 16000
        assert (output / f"episode_{index:06d}.actions.jsonl").is_file()
        sync = json.loads((output / f"audio/episode_{index:06d}.sync.json").read_text())
        assert sync["episode_index"] == index
        assert sync["action_trace"] == f"episode_{index:06d}.actions.jsonl"


def test_legacy_recording_requires_explicit_conversion(tmp_path):
    config = load_config()
    config["dataset"].update(root=str(tmp_path), date="legacy")
    root = tmp_path / "franka_lerobot_legacy"
    (root / "meta").mkdir(parents=True)
    schema = __import__("control.collection.dataset", fromlist=["features"]).features(
        224, "ee", False
    )
    (root / "meta/info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "fps": 30, "features": schema})
    )
    with pytest.raises(ValueError, match="convert a copy"):
        open_dataset(config)


def test_resuming_converted_stats_keeps_metadata_shards_compatible(v3_collection):
    import pyarrow.parquet as pq
    from lerobot.datasets import LeRobotDataset

    config, root = v3_collection
    # Converted v2.1 statistics lack the quantiles generated by a modern writer.
    for path in (root / "meta/episodes").glob("chunk-*/file-*.parquet"):
        table = pq.ParquetFile(path).read()
        columns = [key for key in table.column_names if not key.rsplit("/", 1)[-1].startswith("q")]
        pq.write_table(table.select(columns).replace_schema_metadata(None), path)
    run_collection(config, dry_run=True, max_steps=20)
    dataset = LeRobotDataset(root.name, root=root, video_backend="pyav")
    assert dataset.num_episodes == 4
    paths = list((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    schemas = [pq.read_schema(path) for path in paths]
    assert all(schema.equals(schemas[0], check_metadata=False) for schema in schemas)
