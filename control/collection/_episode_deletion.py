"""Private transactional deletion engine; callers use EpisodeDeleter."""

from __future__ import annotations

import ctypes
import fcntl
import json
import logging
import math
import os
import re
import shutil
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from control.collection.deletion import EpisodeDeletionResult

logger = logging.getLogger(__name__)
_PATH_FIELDS = {"episode_chunk", "episode_index", "chunk_index", "file_index", "video_key"}


def _safe_path(root: Path, relative: str | Path, *, require_file: bool = True) -> Path:
    root = root.resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Dataset metadata contains an unsafe path")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Dataset path escapes its root")
    if require_file and not candidate.is_file():
        raise FileNotFoundError(f"Dataset file not found: {relative}")
    return candidate


def _format_path(root: Path, template: Any, *, require_file: bool = True, **values: Any) -> Path:
    if not isinstance(template, str) or not template:
        raise ValueError("Dataset path template must be a nonempty string")
    try:
        parts = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValueError("Malformed dataset path template") from exc
    if any(
        field is not None and (field not in _PATH_FIELDS or conversion)
        for _, field, _, conversion in parts
    ):
        raise ValueError("Unsupported field in dataset path template")
    try:
        relative = template.format(**values)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Cannot format dataset path from episode metadata") from exc
    return _safe_path(root, relative, require_file=require_file)


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    length: int


@dataclass(frozen=True)
class EpisodeMove:
    old_index: int
    new_index: int
    length: int
    global_start: int


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if text:
        text += "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _is_lerobot_dataset_dir(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _episode_parquet_path(info: Dict[str, Any], dataset_dir: Path, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return _format_path(
        dataset_dir,
        template,
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        require_file=False,
    )


def _episode_video_paths(info: Dict[str, Any], dataset_dir: Path, episode_index: int) -> List[Path]:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size

    videos_dir = dataset_dir / "videos" / f"chunk-{episode_chunk:03d}"
    pattern = f"episode_{episode_index:06d}.mp4"
    paths = set(videos_dir.rglob(pattern)) if videos_dir.exists() else set()
    template = info.get("video_path")
    if template:
        for key, feature in info.get("features", {}).items():
            if feature.get("dtype") == "video":
                path = _format_path(
                    dataset_dir,
                    template,
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                    video_key=key,
                    require_file=False,
                )
                if path.is_file():
                    paths.add(path)
    return sorted(paths)


def _episode_audio_paths(dataset_dir: Path, episode_index: int) -> List[Path]:
    audio_dir = dataset_dir / "audio"
    stem = f"episode_{episode_index:06d}"
    return [
        audio_dir / f"{stem}.wav",
        audio_dir / f"{stem}.audio.json",
        audio_dir / f"{stem}.sync.json",
        audio_dir / "vad_segments" / stem,
        dataset_dir / f"{stem}.sync.json",
        dataset_dir / f"{stem}.actions.jsonl",
    ]


def _episode_audio_path_map(dataset_dir: Path, old_index: int, new_index: int) -> Dict[Path, Path]:
    old_stem = f"episode_{old_index:06d}"
    new_stem = f"episode_{new_index:06d}"
    audio_dir = dataset_dir / "audio"
    return {
        audio_dir / f"{old_stem}.wav": audio_dir / f"{new_stem}.wav",
        audio_dir / f"{old_stem}.audio.json": audio_dir / f"{new_stem}.audio.json",
        audio_dir / f"{old_stem}.sync.json": audio_dir / f"{new_stem}.sync.json",
        audio_dir / "vad_segments" / old_stem: audio_dir / "vad_segments" / new_stem,
        dataset_dir / f"{old_stem}.sync.json": dataset_dir / f"{new_stem}.sync.json",
        dataset_dir / f"{old_stem}.actions.jsonl": dataset_dir / f"{new_stem}.actions.jsonl",
    }


def _update_splits_in_place(
    info: Dict[str, Any], old_total: int, new_total: int, delete_index: int
) -> None:
    splits = info.get("splits")
    if not isinstance(splits, dict):
        return

    for k, v in list(splits.items()):
        if not isinstance(v, str) or ":" not in v:
            continue
        start_s, end_s = v.split(":", 1)
        try:
            start_i = int(start_s)
            end_i = int(end_s)
        except ValueError:
            continue

        if 0 <= start_i <= end_i <= old_total:
            splits[k] = (
                f"{start_i - int(start_i > delete_index)}:{end_i - int(end_i > delete_index)}"
            )


def _load_episode_infos(dataset_dir: Path) -> List[EpisodeInfo]:
    episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    out: List[EpisodeInfo] = []
    for row in episodes:
        if "episode_index" not in row or "length" not in row:
            continue
        try:
            out.append(EpisodeInfo(int(row["episode_index"]), int(row["length"])))
        except (TypeError, ValueError):
            continue
    return sorted(out, key=lambda e: e.episode_index)


def _assert_contiguous_episode_infos(episode_infos: List[EpisodeInfo]) -> None:
    expected = list(range(len(episode_infos)))
    actual = [e.episode_index for e in episode_infos]
    if actual != expected:
        raise RuntimeError(
            "episodes.jsonl 里的 episode_index 不是连续的 0..N-1，拒绝自动重排。\n"
            f"Actual: {actual[:20]}{'...' if len(actual) > 20 else ''}"
        )


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _move_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise RuntimeError(f"目标路径已存在，拒绝覆盖: {dst}")
    shutil.move(str(src), str(dst))


def _replace_column(table: pa.Table, column_name: str, values: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(column_name)
    if idx < 0:
        raise RuntimeError(f"parquet 缺少列: {column_name}")
    return table.set_column(idx, table.schema.field(idx), values)


def _rewrite_parquet_episode(
    *,
    src: Path,
    dst: Path,
    new_episode_index: int,
    global_start: int,
    expected_length: int,
    remove_src: bool = False,
) -> None:
    if not src.exists():
        raise RuntimeError(f"缺少 parquet 文件: {src}")

    table = pq.read_table(src)
    if table.num_rows != expected_length:
        raise RuntimeError(
            f"parquet 行数和 episodes.jsonl length 不一致: {src} "
            f"rows={table.num_rows}, length={expected_length}"
        )

    episode_field = table.schema.field("episode_index")
    index_field = table.schema.field("index")
    episode_values = pa.array([new_episode_index] * table.num_rows, type=episode_field.type)
    index_values = pa.array(
        range(global_start, global_start + table.num_rows), type=index_field.type
    )
    table = _replace_column(table, "episode_index", episode_values)
    table = _replace_column(table, "index", index_values)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, dst)
    if remove_src and src != dst and src.exists():
        src.unlink()


def _episode_video_destination(
    info: Dict[str, Any], dataset_dir: Path, src_video: Path, new_index: int, old_index: int
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    new_chunk = new_index // chunks_size
    template = info.get("video_path")
    if template:
        for key, feature in info.get("features", {}).items():
            if feature.get("dtype") != "video":
                continue
            original = _format_path(
                dataset_dir,
                template,
                episode_chunk=old_index // chunks_size,
                episode_index=old_index,
                video_key=key,
                require_file=False,
            )
            if original == src_video:
                return _format_path(
                    dataset_dir,
                    template,
                    episode_chunk=new_chunk,
                    episode_index=new_index,
                    video_key=key,
                    require_file=False,
                )
    video_key = src_video.parent.name
    return (
        dataset_dir
        / "videos"
        / f"chunk-{new_chunk:03d}"
        / video_key
        / f"episode_{new_index:06d}.mp4"
    )


def _patch_audio_json(path: Path, dataset_dir: Path, episode_index: int) -> None:
    if not path.exists():
        return
    payload = _read_json(path)
    payload["audio_path"] = str(
        (dataset_dir / "audio" / f"episode_{episode_index:06d}.wav").resolve()
    )
    _write_json(path, payload)


def _patch_sync_json(
    path: Path, episode_index: int, old_index: int, final_dataset_dir: Path
) -> None:
    if not path.exists():
        return
    payload = _read_json(path)
    payload["episode_index"] = episode_index
    if "audio_path" in payload:
        payload["audio_path"] = f"audio/episode_{episode_index:06d}.wav"
    if "audio_metadata_path" in payload:
        payload["audio_metadata_path"] = f"audio/episode_{episode_index:06d}.audio.json"
    if "action_trace" in payload:
        payload["action_trace"] = f"episode_{episode_index:06d}.actions.jsonl"
    # Only rewrite identifiers and file references, never transcripts/task labels.
    old_stem = f"episode_{old_index:06d}"
    new_stem = f"episode_{episode_index:06d}"
    path_keys = {
        "audio_path",
        "audio_metadata_path",
        "action_trace",
        "segments_dir",
        "audio_wav_path",
    }

    def patch_references(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                patch_references(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in path_keys | {"seg_id"} and isinstance(item, str):
                    item = re.sub(rf"{old_stem}(?!\d)", new_stem, item)
                    if key in path_keys and Path(item).is_absolute():
                        # Historical metadata can still refer to the old collection root.
                        parts = Path(item).parts
                        if "audio" in parts:
                            item = str(final_dataset_dir.joinpath(*parts[parts.index("audio") :]))
                    value[key] = item
                else:
                    patch_references(item)

    patch_references(payload)
    _write_json(path, payload)


def _patch_stats_row(
    row: Dict[str, Any], *, episode_index: int, global_start: int, length: int
) -> Dict[str, Any]:
    row = dict(row)
    row["episode_index"] = episode_index

    stats = row.get("stats")
    if not isinstance(stats, dict):
        return row

    ep_stats = stats.get("episode_index")
    if isinstance(ep_stats, dict):
        ep_stats["min"] = [episode_index]
        ep_stats["max"] = [episode_index]
        ep_stats["mean"] = [float(episode_index)]
        ep_stats["std"] = [0.0]
        ep_stats["count"] = [length]

    index_stats = stats.get("index")
    if isinstance(index_stats, dict):
        end = global_start + length - 1
        index_stats["min"] = [global_start]
        index_stats["max"] = [end]
        index_stats["mean"] = [float(global_start + end) / 2.0]
        index_stats["std"] = [math.sqrt((length * length - 1) / 12)]
        index_stats["count"] = [length]

    return row


def _aggregate_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool per-episode population statistics without decoding images/video."""
    features = {key for row in rows for key in row.get("stats", {})}
    result = {}
    for key in features:
        values = [row["stats"][key] for row in rows if key in row.get("stats", {})]
        if not all({"min", "max", "mean", "std", "count"} <= value.keys() for value in values):
            raise ValueError("episode stats 缺少聚合字段")
        counts = np.array([np.asarray(value["count"]).item() for value in values], dtype=float)
        if not np.isfinite(counts).all() or np.any(counts <= 0):
            raise ValueError("episode stats count 无效")
        means = np.asarray([value["mean"] for value in values], dtype=float)
        stds = np.asarray([value["std"] for value in values], dtype=float)
        weights = counts.reshape((-1,) + (1,) * (means.ndim - 1))
        mean = (means * weights).sum(axis=0) / counts.sum()
        variance = ((stds**2 + (means - mean) ** 2) * weights).sum(axis=0) / counts.sum()
        result[key] = {
            "min": np.min([value["min"] for value in values], axis=0).tolist(),
            "max": np.max([value["max"] for value in values], axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [int(counts.sum())],
        }
    return result


def _build_moves(episode_infos: List[EpisodeInfo], delete_index: int) -> List[EpisodeMove]:
    moves: List[EpisodeMove] = []
    global_start = 0
    new_index = 0
    for info in episode_infos:
        if info.episode_index == delete_index:
            continue
        moves.append(
            EpisodeMove(
                old_index=info.episode_index,
                new_index=new_index,
                length=info.length,
                global_start=global_start,
            )
        )
        global_start += info.length
        new_index += 1
    return moves


def _rewrite_metadata(
    *,
    dataset_dir: Path,
    info: Dict[str, Any],
    moves: List[EpisodeMove],
    delete_index: int,
    old_total: int,
) -> None:
    meta_dir = dataset_dir / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"
    info_path = meta_dir / "info.json"

    move_by_old = {m.old_index: m for m in moves}

    episodes_rows = _read_jsonl(episodes_path)
    new_episodes_rows: List[Dict[str, Any]] = []
    for row in episodes_rows:
        old_idx = int(row.get("episode_index", -1))
        if old_idx == delete_index:
            continue
        move = move_by_old[old_idx]
        row = dict(row)
        row["episode_index"] = move.new_index
        new_episodes_rows.append(row)
    new_episodes_rows.sort(key=lambda r: int(r["episode_index"]))
    _write_jsonl(episodes_path, new_episodes_rows)

    if stats_path.exists():
        stats_rows = _read_jsonl(stats_path)
        new_stats_rows: List[Dict[str, Any]] = []
        for row in stats_rows:
            old_idx = int(row.get("episode_index", -1))
            if old_idx == delete_index:
                continue
            move = move_by_old[old_idx]
            new_stats_rows.append(
                _patch_stats_row(
                    row,
                    episode_index=move.new_index,
                    global_start=move.global_start,
                    length=move.length,
                )
            )
        new_stats_rows.sort(key=lambda r: int(r["episode_index"]))
        _write_jsonl(stats_path, new_stats_rows)
        if (meta_dir / "stats.json").exists():
            _write_json(meta_dir / "stats.json", _aggregate_stats(new_stats_rows))

    new_total = len(moves)
    chunks_size = int(info.get("chunks_size", 1000))
    info["total_episodes"] = new_total
    info["total_frames"] = sum(m.length for m in moves)
    info["total_chunks"] = (new_total + chunks_size - 1) // chunks_size if new_total > 0 else 0
    if "total_videos" in info:
        info["total_videos"] = sum(
            len(_episode_video_paths(info, dataset_dir, move.new_index)) for move in moves
        )
    _update_splits_in_place(
        info, old_total=old_total, new_total=new_total, delete_index=delete_index
    )
    _write_json(info_path, info)


def _cleanup_empty_dirs(root: Path) -> None:
    for dirname in ("data", "videos"):
        base = root / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass


def _prepare_episode(
    dataset_dir: Path, episode_index: int, *, dry_run: bool, final_dataset_dir: Path | None = None
) -> EpisodeDeletionResult:
    dataset_dir = dataset_dir.resolve()
    final_dataset_dir = final_dataset_dir or dataset_dir
    if not _is_lerobot_dataset_dir(dataset_dir):
        raise FileNotFoundError(f"不是有效的 LeRobot 数据集目录: {dataset_dir}")

    if episode_index < 0:
        raise ValueError("episode_index must be >= 0")

    info = _read_json(dataset_dir / "meta" / "info.json")
    if info.get("codebase_version") == "v3.0":
        from control.collection._v3_deletion import prepare_episode

        return prepare_episode(dataset_dir, episode_index, dry_run, final_dataset_dir)
    if info.get("codebase_version") not in (None, "v2.0", "v2.1"):
        raise ValueError("逐 episode 删除仅支持 v2 数据集")
    # Writes must never follow links outside the selected dataset, including metadata.
    if any(path.is_symlink() for path in dataset_dir.rglob("*")):
        raise ValueError("数据集含符号链接，拒绝删除或重编号")
    episode_infos = _load_episode_infos(dataset_dir)
    _assert_contiguous_episode_infos(episode_infos)

    if episode_index not in {e.episode_index for e in episode_infos}:
        raise ValueError(f"episode_index 不存在: {episode_index}")

    data_paths = [
        _episode_parquet_path(info, dataset_dir, ep.episode_index) for ep in episode_infos
    ]
    if len(set(data_paths)) != len(data_paths):
        raise ValueError("v2 数据路径必须为每个 episode 提供独立文件")
    # Resolve video templates before any deletion, including destinations across chunks.
    seen_videos = set()
    for ep in episode_infos:
        videos = set(_episode_video_paths(info, dataset_dir, ep.episode_index))
        if seen_videos & videos:
            raise ValueError("v2 视频必须为每个 episode 提供独立文件")
        seen_videos.update(videos)
    for ep in episode_infos:
        parquet_path = _episode_parquet_path(info, dataset_dir, ep.episode_index)
        if not parquet_path.exists():
            raise FileNotFoundError(f"缺少 parquet 文件: {parquet_path}")

    old_total = len(episode_infos)
    moves = _build_moves(episode_infos, episode_index)
    # Validate all inputs that will be rewritten before removing any source files.
    for move in moves:
        if move.old_index == move.new_index:
            continue
        table = pq.read_table(_episode_parquet_path(info, dataset_dir, move.old_index))
        if table.num_rows != move.length or not {"episode_index", "index"} <= set(
            table.column_names
        ):
            raise ValueError("Parquet 行数或索引列与元数据不一致，未执行删除")
        for suffix in ("audio.json", "sync.json"):
            for folder in ("audio", ""):
                path = _safe_path(
                    dataset_dir,
                    f"{folder}/episode_{move.old_index:06d}.{suffix}".lstrip("/"),
                    require_file=False,
                )
                if path.exists() and not isinstance(_read_json(path), dict):
                    raise ValueError("Sidecar 必须是 JSON 对象")
    known = {episode.episode_index for episode in episode_infos}
    for filename in ("episodes.jsonl", "episodes_stats.jsonl"):
        if any(
            int(row["episode_index"]) not in known
            for row in _read_jsonl(dataset_dir / "meta" / filename)
        ):
            raise ValueError("元数据包含未知 episode_index")
    if (dataset_dir / "meta" / "stats.json").exists():
        stats_rows = _read_jsonl(dataset_dir / "meta" / "episodes_stats.jsonl")
        move_by_old = {move.old_index: move for move in moves}
        planned_stats = [
            _patch_stats_row(
                row,
                episode_index=move_by_old[int(row["episode_index"])].new_index,
                global_start=move_by_old[int(row["episode_index"])].global_start,
                length=move_by_old[int(row["episode_index"])].length,
            )
            for row in stats_rows
            if int(row["episode_index"]) != episode_index
        ]
        _aggregate_stats(planned_stats)
    deleted = next(ep for ep in episode_infos if ep.episode_index == episode_index)
    result = EpisodeDeletionResult(
        dataset_dir=final_dataset_dir,
        episode_index=episode_index,
        deleted_frames=deleted.length,
        total_episodes_before=len(episode_infos),
        total_episodes_after=len(moves),
        total_frames_before=sum(ep.length for ep in episode_infos),
        total_frames_after=sum(move.length for move in moves),
        dry_run=dry_run,
    )
    if dry_run:
        return result

    # 1) 删除目标 episode 的数据文件。
    _remove_path(_episode_parquet_path(info, dataset_dir, episode_index))
    for path in _episode_video_paths(info, dataset_dir, episode_index):
        _remove_path(path)
    for path in _episode_audio_paths(dataset_dir, episode_index):
        _remove_path(path)

    # 2) 重写并前移后续 episode 的 parquet，同时修正内部 episode_index/index。
    for move in moves:
        if move.old_index == move.new_index:
            continue
        src = _episode_parquet_path(info, dataset_dir, move.old_index)
        dst = _episode_parquet_path(info, dataset_dir, move.new_index)
        _rewrite_parquet_episode(
            src=src,
            dst=dst,
            new_episode_index=move.new_index,
            global_start=move.global_start,
            expected_length=move.length,
            remove_src=True,
        )

    # 3) 前移视频和音频 sidecar，并修正 JSON 中的路径/index。
    for move in moves:
        if move.old_index == move.new_index:
            continue

        for src_video in _episode_video_paths(info, dataset_dir, move.old_index):
            dst_video = _episode_video_destination(
                info, dataset_dir, src_video, move.new_index, move.old_index
            )
            _move_path(src_video, dst_video)

        for src_audio, dst_audio in _episode_audio_path_map(
            dataset_dir, move.old_index, move.new_index
        ).items():
            _move_path(src_audio, dst_audio)

        _patch_audio_json(
            dataset_dir / "audio" / f"episode_{move.new_index:06d}.audio.json",
            final_dataset_dir,
            move.new_index,
        )
        _patch_sync_json(
            dataset_dir / "audio" / f"episode_{move.new_index:06d}.sync.json",
            move.new_index,
            move.old_index,
            final_dataset_dir,
        )
        _patch_sync_json(
            dataset_dir / f"episode_{move.new_index:06d}.sync.json",
            move.new_index,
            move.old_index,
            final_dataset_dir,
        )

    # Canonicalize surviving historical absolute audio paths, including earlier episodes.
    for move in moves:
        if move.old_index != move.new_index:
            continue
        _patch_audio_json(
            dataset_dir / "audio" / f"episode_{move.new_index:06d}.audio.json",
            final_dataset_dir,
            move.new_index,
        )
        for folder in ("audio", ""):
            _patch_sync_json(
                dataset_dir / folder / f"episode_{move.new_index:06d}.sync.json",
                move.new_index,
                move.old_index,
                final_dataset_dir,
            )

    # 4) 更新 meta。
    _rewrite_metadata(
        dataset_dir=dataset_dir,
        info=info,
        moves=moves,
        delete_index=episode_index,
        old_total=old_total,
    )
    _cleanup_empty_dirs(dataset_dir)

    return result


def _source_manifest(root: Path) -> dict:
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("数据集含符号链接，拒绝删除或重编号")
        if path.is_file():
            stat = path.stat()
            result[str(path.relative_to(root))] = (
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
    return result


def _validate_result(root: Path, final_root: Path) -> None:
    info = _read_json(root / "meta/info.json")
    if info["codebase_version"] == "v3.0":
        from data_analysis.dataset_io import episode_dataframe, episode_paths, episode_rows

        episodes = [EpisodeInfo(row["episode_index"], row["length"]) for row in episode_rows(root)]
        paths = episode_paths(root)
    else:
        episodes = _load_episode_infos(root)
    _assert_contiguous_episode_infos(episodes)
    if info["total_episodes"] != len(episodes) or info["total_frames"] != sum(
        ep.length for ep in episodes
    ):
        raise ValueError("删除后的元数据总数不一致")
    start = 0
    for episode in episodes:
        if info["codebase_version"] == "v3.0":
            table = pa.Table.from_pandas(episode_dataframe(paths[episode.episode_index], episode.episode_index))
        else:
            table = pq.read_table(_episode_parquet_path(info, root, episode.episode_index))
        if (
            table.num_rows != episode.length
            or table["episode_index"].to_pylist() != [episode.episode_index] * episode.length
            or table["index"].to_pylist() != list(range(start, start + episode.length))
        ):
            raise ValueError("删除后的 Parquet 行数或索引不一致")
        start += episode.length
        for folder in ("audio", ""):
            sync_path = root / folder / f"episode_{episode.episode_index:06d}.sync.json"
            if not sync_path.exists():
                continue
            payload = _read_json(sync_path)
            if payload.get("episode_index") != episode.episode_index:
                raise ValueError("删除后的同步文件编号不一致")
            references = [
                payload.get(key) for key in ("audio_path", "audio_metadata_path", "action_trace")
            ]
            # Offline VAD records a directory even when zero segments were detected.
            if payload.get("vad_segments"):
                references.append(payload.get("vad_metadata", {}).get("segments_dir"))
            references.extend(
                segment.get("audio_wav_path") for segment in payload.get("vad_segments", [])
            )
            for reference in references:
                if not reference:
                    continue
                path = Path(reference)
                if path.is_absolute():
                    path = path.relative_to(final_root)
                if not _safe_path(root, str(path), require_file=False).exists():
                    raise ValueError(f"删除后的 sidecar 引用不存在: {reference}")
    stats_path = root / "meta/episodes_stats.jsonl"
    if stats_path.exists():
        indices = sorted(int(row["episode_index"]) for row in _read_jsonl(stats_path))
        if indices != list(range(len(episodes))):
            raise ValueError("删除后的 episode stats 缺失或重复")


def _exchange_directories(source: Path, staged: Path) -> None:
    """Linux RENAME_EXCHANGE keeps the dataset path present throughout commit."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("当前系统不支持原子目录交换，原数据未修改")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(staged), 2) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(source))


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _delete_episode(
    dataset_dir: Path, episode_index: int, *, dry_run: bool
) -> EpisodeDeletionResult:
    if dry_run or not _is_lerobot_dataset_dir(dataset_dir):
        return _prepare_episode(dataset_dir, episode_index, dry_run=True)
    # The persistent sibling lock serializes deletion transactions.
    lock_path = dataset_dir.parent / f".{dataset_dir.name}.delete.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("此数据集已有删除任务在运行") from exc
        manifest = _source_manifest(dataset_dir)
        transaction = Path(
            tempfile.mkdtemp(prefix=f".{dataset_dir.name}.delete-", dir=dataset_dir.parent)
        )
        try:
            staged = transaction / "dataset"
            logger.info("[STAGING] 正在复制数据集；原数据保持不变。")
            shutil.copytree(dataset_dir, staged)
            result = _prepare_episode(
                staged, episode_index, dry_run=False, final_dataset_dir=dataset_dir
            )
            _validate_result(staged, dataset_dir)
            # Flush prepared files before publishing the new directory tree.
            for path in staged.rglob("*"):
                if path.is_file():
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
            for path in sorted(staged.rglob("*"), key=lambda path: len(path.parts), reverse=True):
                if path.is_dir():
                    _sync_directory(path)
            _sync_directory(staged)
            _sync_directory(transaction)
            _sync_directory(dataset_dir.parent)
            if manifest != _source_manifest(dataset_dir):
                raise RuntimeError("准备期间原数据集被其他进程修改，已取消提交")
            _exchange_directories(dataset_dir, staged)
            try:
                _sync_directory(dataset_dir.parent)
                _sync_directory(transaction)
            except OSError as exc:
                logger.warning("删除已提交，但目录落盘确认失败: %s", exc)
            logger.info("[OK] 已原子提交删除结果，文件、编号及引用校验通过。")
        finally:
            try:
                shutil.rmtree(transaction)
            except OSError as exc:
                logger.warning("临时目录清理失败，请清理 %s: %s", transaction, exc)
    return result
