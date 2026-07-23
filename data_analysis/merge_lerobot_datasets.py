#!/usr/bin/env python3
"""Merge local LeRobot datasets under data/openpi.

Example:
  uv run data_analysis/merge_lerobot_datasets.py \
    franka_lerobot_4_9_audio franka_lerobot_5_1_audio \
    --output franka_lerobot_mixed_audio

The script creates a new dataset directory and does not modify the inputs.
It rewrites episode_index, frame_index, index, task_index, and audio sidecar
episode numbers so the merged dataset is continuous.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


JSON = dict[str, Any]


@dataclass(frozen=True)
class DatasetSource:
    name: str
    path: Path
    info: JSON
    episodes: list[JSON]
    episode_stats: dict[int, JSON]
    tasks: dict[int, str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_data_root() -> Path:
    return _repo_root() / "data" / "openpi"


def _read_json(path: Path) -> JSON:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[JSON]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [json.loads(line) for line in lines]


def _write_jsonl(path: Path, rows: Iterable[JSON]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (
        path / "meta" / "episodes.jsonl"
    ).is_file()


def _resolve_dataset(name_or_path: str, data_root: Path) -> Path:
    raw = Path(name_or_path).expanduser()
    candidates = []
    if raw.is_absolute() or raw.parts[:1] in ((".",), ("..",)):
        candidates.append(raw)
    else:
        candidates.append(data_root / raw)
        if raw.parts[:1] == ("openpi",):
            candidates.append(data_root.parent / raw)
        candidates.append(raw)

    for candidate in candidates:
        resolved = candidate.resolve()
        if _is_lerobot_dataset(resolved):
            return resolved

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"找不到有效 LeRobot 数据集: {name_or_path} (tried: {tried})"
    )


def _load_tasks(path: Path) -> dict[int, str]:
    tasks_path = path / "meta" / "tasks.jsonl"
    rows = _read_jsonl(tasks_path)
    tasks: dict[int, str] = {}
    for row in rows:
        try:
            tasks[int(row["task_index"])] = str(row["task"])
        except (KeyError, TypeError, ValueError):
            continue
    return tasks


def _load_source(name_or_path: str, data_root: Path) -> DatasetSource:
    path = _resolve_dataset(name_or_path, data_root)
    info = _read_json(path / "meta" / "info.json")
    episodes = _read_jsonl(path / "meta" / "episodes.jsonl")
    stats_rows = _read_jsonl(path / "meta" / "episodes_stats.jsonl")
    episode_stats = {
        int(row["episode_index"]): row for row in stats_rows if "episode_index" in row
    }
    return DatasetSource(
        name=path.name,
        path=path,
        info=info,
        episodes=sorted(episodes, key=lambda row: int(row["episode_index"])),
        episode_stats=episode_stats,
        tasks=_load_tasks(path),
    )


def _episode_chunk(info: JSON, episode_index: int) -> int:
    chunks_size = int(info.get("chunks_size", 1000))
    return int(episode_index) // chunks_size


def _episode_parquet_path(info: JSON, root: Path, episode_index: int) -> Path:
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return root / template.format(
        episode_chunk=_episode_chunk(info, episode_index),
        episode_index=episode_index,
    )


def _episode_video_paths(info: JSON, root: Path, episode_index: int) -> list[Path]:
    template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    chunk = _episode_chunk(info, episode_index)
    videos_dir = root / "videos" / f"chunk-{chunk:03d}"
    if not videos_dir.is_dir():
        return []
    return sorted(videos_dir.rglob(f"episode_{episode_index:06d}.mp4"))


def _validate_compatible(sources: list[DatasetSource]) -> None:
    if not sources:
        raise ValueError("至少需要一个输入数据集")

    base = sources[0]
    base_features = base.info.get("features")
    base_fps = float(base.info.get("fps", 0.0))
    base_robot_type = base.info.get("robot_type")

    for source in sources[1:]:
        if source.info.get("robot_type") != base_robot_type:
            raise ValueError(
                f"robot_type 不一致: {base.name}={base_robot_type}, "
                f"{source.name}={source.info.get('robot_type')}"
            )
        if float(source.info.get("fps", 0.0)) != base_fps:
            raise ValueError(
                f"fps 不一致: {base.name}={base_fps}, "
                f"{source.name}={source.info.get('fps')}"
            )
        if source.info.get("features") != base_features:
            raise ValueError(f"features 不一致，拒绝合并: {base.name} vs {source.name}")


def _task_name(source: DatasetSource, task_index: int, fallback: str = "") -> str:
    if task_index in source.tasks:
        return source.tasks[task_index]
    return fallback


def _episode_tasks(row: JSON) -> list[str]:
    tasks = row.get("tasks")
    if isinstance(tasks, list):
        return [str(task) for task in tasks]
    if isinstance(tasks, str):
        return [tasks]
    return []


def _copy_tree_or_file(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=False)
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_episode_images(
    source_root: Path,
    dest_root: Path,
    old_episode_index: int,
    new_episode_index: int,
) -> None:
    images_dir = source_root / "images"
    if not images_dir.is_dir():
        return

    old_name = f"episode_{old_episode_index:06d}"
    new_name = f"episode_{new_episode_index:06d}"
    for old_dir in images_dir.rglob(old_name):
        if not old_dir.is_dir():
            continue
        rel_parent = old_dir.relative_to(images_dir).parent
        new_dir = dest_root / "images" / rel_parent / new_name
        _copy_tree_or_file(old_dir, new_dir)


def _copy_episode_videos(
    source: DatasetSource,
    dest_root: Path,
    old_episode_index: int,
    new_episode_index: int,
    output_info: JSON,
) -> None:
    old_paths = _episode_video_paths(source.info, source.path, old_episode_index)
    for old_path in old_paths:
        try:
            rel = old_path.relative_to(source.path)
        except ValueError:
            continue
        new_rel = Path(
            str(rel).replace(
                f"episode_{old_episode_index:06d}",
                f"episode_{new_episode_index:06d}",
            )
        )
        new_path = dest_root / new_rel
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, new_path)

    # Keep the destination path convention from the output dataset's info.json.
    _ = output_info


def _rewrite_audio_json(
    path: Path,
    *,
    old_episode_index: int,
    new_episode_index: int,
) -> None:
    try:
        payload = _read_json(path)
    except Exception:
        return

    old_stem = f"episode_{old_episode_index:06d}"
    new_stem = f"episode_{new_episode_index:06d}"
    payload["episode_index"] = new_episode_index

    for key in ("audio_path", "audio_metadata_path"):
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        value = value.replace(old_stem, new_stem)
        if key == "audio_path" and not value.startswith("audio/"):
            value = str((path.parent / f"{new_stem}.wav").resolve())
        elif key == "audio_metadata_path" and not value.startswith("audio/"):
            value = str((path.parent / f"{new_stem}.audio.json").resolve())
        payload[key] = value
    _write_json(path, payload)


def _copy_episode_audio(
    source_root: Path,
    dest_root: Path,
    old_episode_index: int,
    new_episode_index: int,
) -> None:
    audio_dir = source_root / "audio"
    if not audio_dir.is_dir():
        return

    old_stem = f"episode_{old_episode_index:06d}"
    new_stem = f"episode_{new_episode_index:06d}"
    dest_audio_dir = dest_root / "audio"

    for old_path in sorted(audio_dir.glob(f"{old_stem}*")):
        if old_path.is_dir():
            continue
        new_path = dest_audio_dir / old_path.name.replace(old_stem, new_stem, 1)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, new_path)
        if new_path.suffix == ".json":
            _rewrite_audio_json(
                new_path,
                old_episode_index=old_episode_index,
                new_episode_index=new_episode_index,
            )

    vad_dir = audio_dir / "vad_segments" / old_stem
    if vad_dir.exists():
        _copy_tree_or_file(vad_dir, dest_audio_dir / "vad_segments" / new_stem)


def _stat(values: np.ndarray) -> JSON:
    if values.size == 0:
        return {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [0]}
    return {
        "min": [int(values.min())],
        "max": [int(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(values.size)],
    }


def _fix_reindexed_stats(
    row: JSON,
    *,
    new_episode_index: int,
    frame_start: int,
    length: int,
    fps: float,
    task_indices: np.ndarray,
) -> JSON:
    fixed = json.loads(json.dumps(row))
    fixed["episode_index"] = new_episode_index
    stats = fixed.get("stats")
    if not isinstance(stats, dict):
        return fixed

    frame_index = np.arange(length, dtype=np.int64)
    global_index = frame_start + frame_index
    timestamps = frame_index.astype(np.float64) / float(fps)

    stats["episode_index"] = _stat(np.full(length, new_episode_index, dtype=np.int64))
    stats["frame_index"] = _stat(frame_index)
    stats["index"] = _stat(global_index)
    stats["task_index"] = _stat(task_indices.astype(np.int64, copy=False))
    stats["timestamp"] = {
        "min": [float(timestamps.min()) if length else 0.0],
        "max": [float(timestamps.max()) if length else 0.0],
        "mean": [float(timestamps.mean()) if length else 0.0],
        "std": [float(timestamps.std()) if length else 0.0],
        "count": [length],
    }
    return fixed


def _default_output_name(inputs: list[str]) -> str:
    names = [Path(item).name for item in inputs]
    if len(names) == 1:
        return f"{names[0]}_merged"
    if len(names) == 2:
        return f"{names[0]}__plus__{names[1]}"
    return f"{names[0]}__plus_{len(names) - 1}_datasets"


def _column_to_int64_numpy(table: pa.Table, name: str) -> np.ndarray:
    return table.column(name).combine_chunks().to_numpy(
        zero_copy_only=False,
    ).astype(np.int64, copy=False)


def _replace_or_append_column(
    table: pa.Table,
    name: str,
    values: np.ndarray,
    fallback_type: pa.DataType,
) -> pa.Table:
    field_index = table.schema.get_field_index(name)
    if field_index >= 0:
        field = table.schema.field(field_index)
    else:
        field = pa.field(name, fallback_type)

    array = pa.array(values, type=field.type)
    if field_index >= 0:
        return table.set_column(field_index, field, array)
    return table.append_column(field, array)


def merge_datasets(
    input_names: list[str],
    *,
    output_name: str | None,
    data_root: Path,
    overwrite: bool,
) -> Path:
    sources = [_load_source(name, data_root) for name in input_names]
    _validate_compatible(sources)

    dest_root = data_root / (output_name or _default_output_name(input_names))
    if dest_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"输出目录已存在: {dest_root}。换 --output 名字，或显式加 --overwrite。"
            )
        shutil.rmtree(dest_root)

    fps = float(sources[0].info["fps"])
    output_info = json.loads(json.dumps(sources[0].info))
    output_info["splits"] = {"train": "0:0"}

    dest_meta_dir = dest_root / "meta"
    dest_meta_dir.mkdir(parents=True, exist_ok=False)

    task_to_new_index: dict[str, int] = {}
    episodes_out: list[JSON] = []
    stats_out: list[JSON] = []
    manifest_rows: list[JSON] = []
    total_frames = 0
    new_episode_index = 0

    for source in sources:
        print(f"[Source] {source.name}: {len(source.episodes)} episodes")
        for episode_row in source.episodes:
            old_episode_index = int(episode_row["episode_index"])
            src_parquet = _episode_parquet_path(
                source.info,
                source.path,
                old_episode_index,
            )
            if not src_parquet.is_file():
                raise FileNotFoundError(f"缺少 episode parquet: {src_parquet}")

            table = pq.read_table(src_parquet)
            length = table.num_rows
            if length <= 0:
                print(f"[Skip] empty episode: {source.name}/{old_episode_index}")
                continue

            episode_tasks = _episode_tasks(episode_row)
            fallback_task = episode_tasks[0] if episode_tasks else ""
            if "task_index" in table.column_names:
                old_task_indices = _column_to_int64_numpy(table, "task_index")
                new_task_indices = np.empty_like(old_task_indices)
                old_task_to_name: dict[int, str] = {}
                for old_task_index in sorted(set(old_task_indices.tolist())):
                    task = _task_name(source, int(old_task_index), fallback_task)
                    old_task_to_name[int(old_task_index)] = task
                    if task not in task_to_new_index:
                        task_to_new_index[task] = len(task_to_new_index)
                    new_task_indices[old_task_indices == old_task_index] = (
                        task_to_new_index[task]
                    )
            else:
                task = fallback_task
                if task not in task_to_new_index:
                    task_to_new_index[task] = len(task_to_new_index)
                new_task_indices = np.full(
                    length, task_to_new_index[task], dtype=np.int64
                )
                old_task_to_name = {0: task}

            table = _replace_or_append_column(
                table,
                "episode_index",
                np.full(length, new_episode_index, dtype=np.int64),
                pa.int64(),
            )
            table = _replace_or_append_column(
                table,
                "frame_index",
                np.arange(length, dtype=np.int64),
                pa.int64(),
            )
            table = _replace_or_append_column(
                table,
                "index",
                np.arange(total_frames, total_frames + length, dtype=np.int64),
                pa.int64(),
            )
            table = _replace_or_append_column(
                table,
                "task_index",
                new_task_indices.astype(np.int64, copy=False),
                pa.int64(),
            )
            if "timestamp" in table.column_names:
                table = _replace_or_append_column(
                    table,
                    "timestamp",
                    np.arange(length, dtype=np.float32) / np.float32(fps),
                    pa.float32(),
                )

            dst_parquet = _episode_parquet_path(
                output_info,
                dest_root,
                new_episode_index,
            )
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dst_parquet)

            new_episode_tasks = list(dict.fromkeys(old_task_to_name.values()))
            if episode_tasks:
                new_episode_tasks = episode_tasks

            new_episode_row = json.loads(json.dumps(episode_row))
            new_episode_row["episode_index"] = new_episode_index
            new_episode_row["length"] = length
            new_episode_row["tasks"] = new_episode_tasks
            episodes_out.append(new_episode_row)

            stats_row = source.episode_stats.get(old_episode_index)
            if stats_row is not None:
                stats_out.append(
                    _fix_reindexed_stats(
                        stats_row,
                        new_episode_index=new_episode_index,
                        frame_start=total_frames,
                        length=length,
                        fps=fps,
                        task_indices=new_task_indices,
                    )
                )

            _copy_episode_audio(
                source.path,
                dest_root,
                old_episode_index,
                new_episode_index,
            )
            _copy_episode_images(
                source.path,
                dest_root,
                old_episode_index,
                new_episode_index,
            )
            _copy_episode_videos(
                source,
                dest_root,
                old_episode_index,
                new_episode_index,
                output_info,
            )

            manifest_rows.append(
                {
                    "source_dataset": source.name,
                    "source_path": str(source.path),
                    "source_episode_index": old_episode_index,
                    "merged_episode_index": new_episode_index,
                    "frame_start": total_frames,
                    "length": length,
                }
            )
            total_frames += length
            new_episode_index += 1

    total_episodes = len(episodes_out)
    chunks_size = int(output_info.get("chunks_size", 1000))
    output_info["total_episodes"] = total_episodes
    output_info["total_frames"] = total_frames
    output_info["total_tasks"] = len(task_to_new_index)
    output_info["total_chunks"] = (
        (total_episodes + chunks_size - 1) // chunks_size if total_episodes else 0
    )
    output_info["total_videos"] = len(list((dest_root / "videos").rglob("*.mp4")))
    output_info["splits"] = {"train": f"0:{total_episodes}"}

    task_rows = [
        {"task_index": idx, "task": task}
        for task, idx in sorted(task_to_new_index.items(), key=lambda item: item[1])
    ]

    _write_json(dest_meta_dir / "info.json", output_info)
    _write_jsonl(dest_meta_dir / "tasks.jsonl", task_rows)
    _write_jsonl(dest_meta_dir / "episodes.jsonl", episodes_out)
    _write_jsonl(dest_meta_dir / "episodes_stats.jsonl", stats_out)
    _write_json(
        dest_meta_dir / "merge_manifest.json",
        {
            "sources": [
                {"name": source.name, "path": str(source.path)} for source in sources
            ],
            "episodes": manifest_rows,
        },
    )

    print("=" * 70)
    print(f"[OK] merged dataset: {dest_root}")
    print(f"Episodes: {total_episodes}")
    print(f"Frames: {total_frames}")
    print(f"Tasks: {len(task_rows)}")
    print("=" * 70)
    return dest_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="合并本地 data/openpi 下的 LeRobot 数据集"
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="输入数据集名或路径，例如 franka_lerobot_4_9_audio",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="输出数据集名；默认使用 <第一个>__plus__<第二个>",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_default_data_root(),
        help="包含数据集目录的根目录，默认 data/openpi",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出目录已存在，先删除再重新生成",
    )
    args = parser.parse_args(argv)

    if len(args.datasets) < 2:
        print("[ERROR] 至少传入两个数据集名。", file=sys.stderr)
        return 2

    try:
        merge_datasets(
            args.datasets,
            output_name=args.output,
            data_root=args.data_root.resolve(),
            overwrite=bool(args.overwrite),
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
