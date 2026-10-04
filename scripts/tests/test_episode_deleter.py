"""Deletion preserves surviving samples and publishes a complete tree or no change."""

import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from control.collection import EpisodeDeleter
from control.collection import _episode_deletion as deletion


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def snapshot(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def numeric_stats(values):
    values = np.asarray(values)
    return {
        "min": [float(values.min())],
        "max": [float(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [len(values)],
    }


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    episodes, stats, all_values = [], [], []
    start = 0
    for index, length in enumerate((2, 3, 4, 5)):
        values = list(range(index * 10, index * 10 + length))
        global_indices = list(range(start, start + length))
        table = pa.table(
            {
                "episode_index": pa.array([index] * length, type=pa.int64()),
                "index": pa.array(global_indices, type=pa.int64()),
                "frame_index": pa.array(range(length), type=pa.int64()),
                "timestamp": [frame / 20 for frame in range(length)],
                "action": values,
                "image": [
                    {"bytes": bytes([index, frame]), "path": None} for frame in range(length)
                ],
            }
        )
        parquet = root / f"data/chunk-{index // 2:03d}/episode_{index:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet)
        video = root / f"videos/chunk-{index // 2:03d}/camera/episode_{index:06d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(bytes([index]) * 10)
        audio = root / f"audio/episode_{index:06d}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(bytes([index]) * 12)
        clips = root / f"audio/vad_segments/episode_{index:06d}"
        clips.mkdir(parents=True)
        (clips / "seg_000.wav").write_bytes(bytes([index]) * 4)
        write_json(
            root / f"audio/episode_{index:06d}.audio.json",
            {
                "audio_path": f"/obsolete/dataset/audio/episode_{index:06d}.wav",
                "marker": index,
            },
        )
        write_json(
            root / f"audio/episode_{index:06d}.sync.json",
            {
                "episode_index": index,
                "audio_path": f"audio/episode_{index:06d}.wav",
                "audio_metadata_path": f"audio/episode_{index:06d}.audio.json",
                "vad_metadata": {"segments_dir": f"audio/vad_segments/episode_{index:06d}"},
                "vad_segments": [
                    {
                        "seg_id": f"episode_{index:06d}_vad_000",
                        "audio_wav_path": str(clips / "seg_000.wav")
                        if index % 2
                        else str(clips.relative_to(root) / "seg_000.wav"),
                        "asr_transcript": f"episode_{index:06d} should remain in this transcript",
                        "verified": True,
                    }
                ],
            },
        )
        (root / f"episode_{index:06d}.actions.jsonl").write_text(
            json.dumps({"marker": index}) + "\n"
        )
        write_json(
            root / f"episode_{index:06d}.sync.json",
            {
                "episode_index": index,
                "action_trace": f"episode_{index:06d}.actions.jsonl",
            },
        )
        episodes.append({"episode_index": index, "length": length, "tasks": [f"task {index}"]})
        stats.append(
            {
                "episode_index": index,
                "stats": {
                    "episode_index": numeric_stats([index] * length),
                    "index": numeric_stats(global_indices),
                    "action": numeric_stats(values),
                },
            }
        )
        all_values.extend(values)
        start += length
    write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v2.1",
            "chunks_size": 2,
            "total_episodes": 4,
            "total_frames": start,
            "total_chunks": 2,
            "total_videos": 4,
            "splits": {"train": "0:2", "validation": "2:4"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"camera": {"dtype": "video"}},
        },
    )
    write_jsonl(root / "meta/episodes.jsonl", episodes)
    write_jsonl(root / "meta/episodes_stats.jsonl", stats)
    write_json(root / "meta/stats.json", {"action": numeric_stats(all_values)})
    return root


@pytest.mark.parametrize("deleted", [0, 1, 3])
def test_delete_preserves_contents_across_chunks_and_updates_references(dataset, deleted):
    deleter = EpisodeDeleter(str(dataset))
    before = snapshot(dataset)
    preview = deleter.preview_episode(deleted)
    assert preview.dry_run
    assert preview.episode_index == deleted
    assert preview.dataset_dir == dataset
    assert preview.total_episodes_before == 4
    assert preview.total_episodes_after == 3
    assert snapshot(dataset) == before
    original = [
        pq.read_table(dataset / f"data/chunk-{i // 2:03d}/episode_{i:06d}.parquet")
        for i in range(4)
    ]
    result = deleter.delete_episode(deleted)
    assert not result.dry_run
    assert result.dataset_dir == dataset
    assert result.total_episodes_after == preview.total_episodes_after
    assert result.deleted_frames == preview.deleted_frames
    info = json.loads((dataset / "meta/info.json").read_text())
    assert info["total_episodes"] == 3 and info["total_videos"] == 3 and info["total_chunks"] == 2
    assert info["splits"] == {
        "train": f"0:{2 - int(deleted < 2)}",
        "validation": f"{2 - int(deleted < 2)}:3",
    }
    start, actions = 0, []
    for new, old in enumerate(i for i in range(4) if i != deleted):
        actual = pq.read_table(dataset / f"data/chunk-{new // 2:03d}/episode_{new:06d}.parquet")
        for key in actual.column_names:
            if key not in ("episode_index", "index"):
                assert actual[key].equals(original[old][key]), key
        assert actual["episode_index"].to_pylist() == [new] * actual.num_rows
        assert actual["index"].to_pylist() == list(range(start, start + actual.num_rows))
        start += actual.num_rows
        actions.extend(actual["action"].to_pylist())
        for relative in (
            f"videos/chunk-{new // 2:03d}/camera/episode_{new:06d}.mp4",
            f"audio/episode_{new:06d}.wav",
            f"episode_{new:06d}.actions.jsonl",
        ):
            old_relative = relative.replace(f"episode_{new:06d}", f"episode_{old:06d}")
            if relative.startswith("videos"):
                old_relative = old_relative.replace(
                    f"chunk-{new // 2:03d}", f"chunk-{old // 2:03d}"
                )
            assert (
                hashlib.sha256((dataset / relative).read_bytes()).hexdigest()
                == before[old_relative]
            )
        sync = json.loads((dataset / f"audio/episode_{new:06d}.sync.json").read_text())
        assert sync["episode_index"] == new
        segment = sync["vad_segments"][0]
        assert segment["seg_id"] == f"episode_{new:06d}_vad_000"
        assert segment["asr_transcript"] == f"episode_{old:06d} should remain in this transcript"
        assert segment["verified"] is True
        for reference in (
            sync["audio_path"],
            sync["audio_metadata_path"],
            sync["vad_metadata"]["segments_dir"],
            segment["audio_wav_path"],
        ):
            path = Path(reference)
            assert (path if path.is_absolute() else dataset / path).exists()
        audio_meta = json.loads((dataset / f"audio/episode_{new:06d}.audio.json").read_text())
        assert audio_meta["audio_path"] == str(dataset / f"audio/episode_{new:06d}.wav")
        root_sync = json.loads((dataset / f"episode_{new:06d}.sync.json").read_text())
        assert root_sync["episode_index"] == new
        assert (dataset / root_sync["action_trace"]).is_file()
    assert info["total_frames"] == start
    global_stats = json.loads((dataset / "meta/stats.json").read_text())["action"]
    for key, expected in numeric_stats(actions).items():
        np.testing.assert_allclose(global_stats[key], expected)
    assert len(list(dataset.rglob("*.parquet"))) == 3
    assert not list(dataset.parent.glob(f".{dataset.name}.delete-*"))


@pytest.mark.parametrize(
    "failure", ["copy", "parquet", "metadata", "validation", "fsync", "commit", "interrupt"]
)
def test_failure_leaves_original_tree_byte_identical(dataset, monkeypatch, failure):
    before = snapshot(dataset)

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    if failure == "copy":
        monkeypatch.setattr(deletion.shutil, "copytree", fail)
    elif failure == "parquet":
        monkeypatch.setattr(deletion.pq, "write_table", fail)
    elif failure == "metadata":
        original = deletion._write_jsonl

        def fail_metadata(path, rows):
            original(path, rows)
            if path.name == "episodes_stats.jsonl":
                fail()

        monkeypatch.setattr(deletion, "_write_jsonl", fail_metadata)
    elif failure == "validation":
        monkeypatch.setattr(deletion, "_validate_result", fail)
    elif failure == "fsync":
        monkeypatch.setattr(deletion.os, "fsync", fail)
    elif failure == "interrupt":

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(deletion.pq, "write_table", interrupt)
    else:
        monkeypatch.setattr(deletion, "_exchange_directories", fail)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else OSError):
        EpisodeDeleter(dataset).delete_episode(1)
    assert snapshot(dataset) == before
    assert not list(dataset.parent.glob(f".{dataset.name}.delete-*"))


def test_concurrent_source_change_cancels_commit(dataset, monkeypatch):
    validate = deletion._validate_result

    def change_source(*args):
        validate(*args)
        (dataset / "new-record.txt").write_text("new data")

    monkeypatch.setattr(deletion, "_validate_result", change_source)
    before = snapshot(dataset)
    with pytest.raises(RuntimeError, match="其他进程修改"):
        EpisodeDeleter(dataset).delete_episode(1)
    after = snapshot(dataset)
    assert after.pop("new-record.txt")
    assert after == before


def test_delete_until_empty(dataset):
    for index in (1, 2, 0, 0):
        assert EpisodeDeleter(dataset).delete_episode(index).episode_index == index
    info = json.loads((dataset / "meta/info.json").read_text())
    assert (
        info["total_frames"]
        == info["total_episodes"]
        == info["total_videos"]
        == info["total_chunks"]
        == 0
    )
    assert not (dataset / "meta/episodes.jsonl").read_text()
    assert not list(dataset.rglob("*.parquet"))
    assert not list(dataset.rglob("*.wav"))


def test_missing_vad_reference_refuses_commit(dataset):
    sync_path = dataset / "audio/episode_000002.sync.json"
    sync = json.loads(sync_path.read_text())
    sync["vad_segments"][0]["audio_wav_path"] = "audio/vad_segments/episode_000002/missing.wav"
    write_json(sync_path, sync)
    before = snapshot(dataset)
    with pytest.raises(ValueError, match="引用不存在"):
        EpisodeDeleter(dataset).delete_episode(1)
    assert snapshot(dataset) == before


def test_empty_vad_does_not_require_a_clip_directory(dataset):
    sync_path = dataset / "audio/episode_000002.sync.json"
    sync = json.loads(sync_path.read_text())
    sync["vad_segments"] = []
    write_json(sync_path, sync)
    shutil.rmtree(dataset / "audio/vad_segments/episode_000002")
    assert EpisodeDeleter(dataset).delete_episode(1).episode_index == 1


def test_another_deletion_lock_refuses_to_start(dataset):
    before = snapshot(dataset)
    with (dataset.parent / f".{dataset.name}.delete.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="已有删除任务"):
            EpisodeDeleter(dataset).delete_episode(1)
    assert snapshot(dataset) == before


def test_deleter_reuses_dataset_after_directory_exchange(dataset):
    deleter = EpisodeDeleter(dataset)
    assert deleter.delete_episode(1).total_episodes_after == 3
    # The same instance must use fresh metadata and the new directory tree.
    assert deleter.delete_episode(2).total_episodes_after == 2
    info = json.loads((dataset / "meta/info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 6


@pytest.mark.parametrize("episode_index", [-1, 99])
def test_deleter_invalid_index_does_not_modify_dataset(dataset, episode_index):
    before = snapshot(dataset)
    with pytest.raises(ValueError):
        EpisodeDeleter(dataset).delete_episode(episode_index)
    assert snapshot(dataset) == before
    assert not list(dataset.parent.glob(f".{dataset.name}.delete-*"))


@pytest.mark.parametrize("episode_index", [True, 1.5, "1"])
def test_deleter_rejects_non_integer_index(dataset, episode_index):
    before = snapshot(dataset)
    with pytest.raises(TypeError, match="episode_index"):
        EpisodeDeleter(dataset).delete_episode(episode_index)
    assert snapshot(dataset) == before


def test_deleter_rejects_non_boolean_dry_run(dataset):
    before = snapshot(dataset)
    with pytest.raises(TypeError, match="dry_run"):
        EpisodeDeleter(dataset).delete_episode(1, dry_run="false")
    assert snapshot(dataset) == before


def test_killed_process_before_commit_leaves_original_intact(dataset):
    before = snapshot(dataset)
    program = """
import os, signal, sys
from pathlib import Path
from control.collection import EpisodeDeleter
from control.collection import _episode_deletion as deletion
def kill_before_commit(*args):
    os.kill(os.getpid(), signal.SIGKILL)
deletion._exchange_directories = kill_before_commit
EpisodeDeleter(Path(sys.argv[1])).delete_episode(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(dataset)], capture_output=True, timeout=10
    )
    assert result.returncode == -9
    assert snapshot(dataset) == before
    # A forced kill can leave staging files, but they are outside the live dataset.
    assert list(dataset.parent.glob(f".{dataset.name}.delete-*"))
    assert (dataset / "meta/info.json").is_file()


def test_empty_dataset_raises_instead_of_reporting_a_deletion(dataset):
    deleter = EpisodeDeleter(dataset)
    for index in (1, 2, 0, 0):
        deleter.delete_episode(index)
    before = snapshot(dataset)
    with pytest.raises(ValueError, match="不存在"):
        deleter.delete_episode(0)
    assert snapshot(dataset) == before


def test_missing_dataset_raises_without_creating_files(tmp_path):
    root = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        EpisodeDeleter(root).delete_episode(0)
    assert not list(tmp_path.iterdir())


def test_cli_previews_then_deletes_through_public_class(dataset, capsys):
    from control.collection.deletion import cli

    args = ["--dataset", str(dataset), "--episode-index", "1"]
    before = snapshot(dataset)
    assert cli(args) == 0
    assert "[DRY-RUN]" in capsys.readouterr().out
    assert snapshot(dataset) == before
    assert cli([*args, "--yes"]) == 0
    assert "Total episodes: 4 -> 3" in capsys.readouterr().out
    assert json.loads((dataset / "meta/info.json").read_text())["total_episodes"] == 3


def test_cli_translates_invalid_input_to_exit_code(dataset, capsys):
    from control.collection.deletion import cli

    before = snapshot(dataset)
    assert cli(["--dataset", str(dataset), "--episode-index", "99", "--yes"]) == 2
    assert "[ERROR]" in capsys.readouterr().err
    assert snapshot(dataset) == before
