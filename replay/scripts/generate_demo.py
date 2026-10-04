"""Generate matching LeRobot v2.1/v3.0 fixtures with browser-playable H264 cameras.

Run from repository root: uv run python -m replay.scripts.generate_demo
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa

from ._demo import (
    _CAMERAS,
    _EPISODES,
    _FPS,
    _FRAMES,
    _MARKER,
    _MARKER_TEXT,
    _TASKS,
    _aggregate_video_stats,
    _concat_videos,
    _encode_episode,
    _info,
    _signals,
    _stats,
    _table,
    _tasks_table,
    _write_json,
    _write_jsonl,
    _write_table,
)


def generate_demo(output: Path | None = None) -> list[Path]:
    """Build 3 x 6-second episodes in both formats, replacing only marked demo datasets.

    Existing unmarked datasets and symlink targets are refused. New fixtures are
    staged completely before replacing prior generated targets.
    """
    output = Path(output or Path(__file__).resolve().parents[1] / "demo_data").resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / "demo_v21", output / "demo_v30"]
    for target in targets:
        if target.is_symlink():
            raise ValueError("Refusing to overwrite a symlink demo target")
        if target.exists() and (
            not (target / _MARKER).is_file()
            or (target / _MARKER).is_symlink()
            or (target / _MARKER).read_text() != _MARKER_TEXT
        ):
            raise ValueError(f"Refusing to overwrite unmarked dataset: {target.name}")
    with tempfile.TemporaryDirectory(prefix=".replay-build-", dir=output) as temporary:
        v21, v30 = [Path(temporary) / name for name in ("demo_v21", "demo_v30")]
        signals = [_signals(episode) for episode in range(_EPISODES)]
        episode_stats = [_stats(values) for values in signals]
        summaries = [
            {"episode_index": episode, "tasks": [_TASKS[episode]], "length": _FRAMES}
            for episode in range(_EPISODES)
        ]
        for episode in range(_EPISODES):
            _write_table(
                v21 / f"data/chunk-000/episode_{episode:06d}.parquet", _table(signals[episode])
            )
            for camera in _CAMERAS:
                episode_stats[episode][camera] = _encode_episode(
                    v21 / f"videos/chunk-000/{camera}/episode_{episode:06d}.mp4",
                    episode,
                    camera,
                    signals[episode],
                )
        global_signals = {
            key: np.concatenate([values[key] for values in signals]) for key in signals[0]
        }
        global_stats = _stats(global_signals)
        for camera in _CAMERAS:
            global_stats[camera] = _aggregate_video_stats(episode_stats, camera)
            _concat_videos(
                [
                    v21 / f"videos/chunk-000/{camera}/episode_{episode:06d}.mp4"
                    for episode in range(_EPISODES)
                ],
                v30 / f"videos/{camera}/chunk-000/file-000.mp4",
            )
        _write_table(v30 / "data/chunk-000/file-000.parquet", _table(global_signals))
        packed_episodes = []
        for episode, summary in enumerate(summaries):
            metadata = {
                **summary,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": episode * _FRAMES,
                "dataset_to_index": (episode + 1) * _FRAMES,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            for camera in _CAMERAS:
                metadata.update(
                    {
                        f"videos/{camera}/chunk_index": 0,
                        f"videos/{camera}/file_index": 0,
                        f"videos/{camera}/from_timestamp": episode * _FRAMES / _FPS,
                        f"videos/{camera}/to_timestamp": (episode + 1) * _FRAMES / _FPS,
                    }
                )
            for key, statistics in episode_stats[episode].items():
                for statistic, value in statistics.items():
                    metadata[f"stats/{key}/{statistic}"] = value
            packed_episodes.append(metadata)
        _write_jsonl(v21 / "meta/episodes.jsonl", summaries)
        _write_jsonl(
            v21 / "meta/episodes_stats.jsonl",
            [
                {"episode_index": episode, "stats": statistics}
                for episode, statistics in enumerate(episode_stats)
            ],
        )
        _write_table(
            v30 / "meta/episodes/chunk-000/file-000.parquet", pa.Table.from_pylist(packed_episodes)
        )
        task_records = [
            {"task_index": episode, "task": task} for episode, task in enumerate(_TASKS)
        ]
        for root, version in ((v21, "v2.1"), (v30, "v3.0")):
            _write_json(root / "meta/info.json", _info(version))
            _write_json(root / "meta/stats.json", global_stats)
            _write_jsonl(root / "meta/tasks.jsonl", task_records)
            (root / _MARKER).write_text(_MARKER_TEXT)
        # Current LeRobot v3 releases use tasks.parquet; keep JSONL for older v3 readers too.
        _write_table(v30 / "meta/tasks.parquet", _tasks_table())
        for source, target in zip((v21, v30), targets, strict=True):
            if target.exists():
                shutil.rmtree(target)
            source.rename(target)
    return targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Parent directory for both generated datasets")
    arguments = parser.parse_args()
    for dataset in generate_demo(arguments.output):
        print(f"Created {dataset} (3 episodes, 540 frames, 2 H264 cameras)")
