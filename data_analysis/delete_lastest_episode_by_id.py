"""按 episode_index 删除 LeRobot 数据集中的一个 episode，并重排后续编号。

默认只做 dry-run，不会修改任何文件。确认预览无误后加 --yes 才会硬删除。

会同步处理：
- data/chunk-*/episode_*.parquet
- videos/chunk-*/*/episode_*.mp4（如果存在）
- audio/episode_*.wav、audio/episode_*.audio.json、audio/episode_*.sync.json
- audio/vad_segments/episode_*（如果存在）
- meta/episodes.jsonl、meta/episodes_stats.jsonl、meta/info.json

用法示例：
  uv run data_analysis/delete_lastest_episode_by_id.py \
    --dataset data/openpi/franka_lerobot_4_9_audio \
    --episode-index 12

  uv run data_analysis/delete_lastest_episode_by_id.py \
    --dataset data/openpi/franka_lerobot_4_9_audio \
    --episode-index 12 \
    --yes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


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
    tmp.write_text(
        json.dumps(obj, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )
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
    return (path / "meta" / "info.json").exists() and (
        path / "meta" / "episodes.jsonl"
    ).exists()


def _find_latest_dataset_dir(root: Path) -> Optional[Path]:
    candidates: List[Tuple[float, Path]] = []
    for info_json in root.rglob("meta/info.json"):
        dataset_dir = info_json.parent.parent
        if not _is_lerobot_dataset_dir(dataset_dir):
            continue
        try:
            mtime = info_json.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, dataset_dir))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _episode_parquet_path(
    info: Dict[str, Any], dataset_dir: Path, episode_index: int
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    rel = template.format(episode_chunk=episode_chunk, episode_index=episode_index)
    return dataset_dir / rel


def _episode_video_paths(
    info: Dict[str, Any], dataset_dir: Path, episode_index: int
) -> List[Path]:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size

    videos_dir = dataset_dir / "videos" / f"chunk-{episode_chunk:03d}"
    if not videos_dir.exists():
        return []

    pattern = f"episode_{episode_index:06d}.mp4"
    return sorted(videos_dir.rglob(pattern))


def _episode_audio_paths(dataset_dir: Path, episode_index: int) -> List[Path]:
    audio_dir = dataset_dir / "audio"
    if not audio_dir.exists():
        return []

    stem = f"episode_{episode_index:06d}"
    return [
        audio_dir / f"{stem}.wav",
        audio_dir / f"{stem}.audio.json",
        audio_dir / f"{stem}.sync.json",
        audio_dir / "vad_segments" / stem,
    ]


def _episode_audio_path_map(
    dataset_dir: Path, old_index: int, new_index: int
) -> Dict[Path, Path]:
    old_stem = f"episode_{old_index:06d}"
    new_stem = f"episode_{new_index:06d}"
    audio_dir = dataset_dir / "audio"
    return {
        audio_dir / f"{old_stem}.wav": audio_dir / f"{new_stem}.wav",
        audio_dir / f"{old_stem}.audio.json": audio_dir / f"{new_stem}.audio.json",
        audio_dir / f"{old_stem}.sync.json": audio_dir / f"{new_stem}.sync.json",
        audio_dir / "vad_segments" / old_stem: audio_dir
        / "vad_segments"
        / new_stem,
    }


def _update_splits_in_place(
    info: Dict[str, Any], old_total: int, new_total: int
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

        if end_i == old_total:
            splits[k] = f"{start_i}:{new_total}"


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
    episode_values = pa.array(
        [new_episode_index] * table.num_rows, type=episode_field.type
    )
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
    info: Dict[str, Any], dataset_dir: Path, src_video: Path, new_index: int
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    new_chunk = new_index // chunks_size
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


def _patch_sync_json(path: Path, episode_index: int) -> None:
    if not path.exists():
        return
    payload = _read_json(path)
    payload["episode_index"] = episode_index
    payload["audio_path"] = f"audio/episode_{episode_index:06d}.wav"
    payload["audio_metadata_path"] = f"audio/episode_{episode_index:06d}.audio.json"
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
        frame_stats = stats.get("frame_index")
        frame_std = None
        if isinstance(frame_stats, dict):
            std = frame_stats.get("std")
            if isinstance(std, list) and std:
                frame_std = std[0]
        index_stats["min"] = [global_start]
        index_stats["max"] = [end]
        index_stats["mean"] = [float(global_start + end) / 2.0]
        index_stats["std"] = [frame_std if frame_std is not None else 0.0]
        index_stats["count"] = [length]

    return row


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

    new_total = len(moves)
    chunks_size = int(info.get("chunks_size", 1000))
    info["total_episodes"] = new_total
    info["total_frames"] = sum(m.length for m in moves)
    info["total_chunks"] = (
        (new_total + chunks_size - 1) // chunks_size if new_total > 0 else 0
    )
    _update_splits_in_place(info, old_total=old_total, new_total=new_total)
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


def _print_plan(
    *,
    dataset_dir: Path,
    info: Dict[str, Any],
    episode_infos: List[EpisodeInfo],
    delete_index: int,
    moves: List[EpisodeMove],
) -> None:
    delete_info = next(e for e in episode_infos if e.episode_index == delete_index)
    old_total = len(episode_infos)
    new_total = len(moves)
    old_frames = sum(e.length for e in episode_infos)
    new_frames = sum(m.length for m in moves)

    print(f"Dataset: {dataset_dir}")
    print(f"Delete episode_index: {delete_index}")
    print(f"Deleted frames: {delete_info.length}")
    print(f"Total episodes: {old_total} -> {new_total}")
    print(f"Total frames: {old_frames} -> {new_frames}")
    print(f"Remove parquet: {_episode_parquet_path(info, dataset_dir, delete_index)}")

    video_paths = _episode_video_paths(info, dataset_dir, delete_index)
    audio_paths = [p for p in _episode_audio_paths(dataset_dir, delete_index) if p.exists()]
    print(f"Remove videos: {len(video_paths)}")
    print(f"Remove audio/sidecars: {len(audio_paths)}")

    renumbered = [m for m in moves if m.old_index != m.new_index]
    if renumbered:
        first = renumbered[0]
        last = renumbered[-1]
        print(
            "Renumber episodes: "
            f"{first.old_index:06d}->{first.new_index:06d} ... "
            f"{last.old_index:06d}->{last.new_index:06d}"
        )
    else:
        print("Renumber episodes: none (deleting latest episode)")


def delete_episode_by_id(dataset_dir: Path, episode_index: int, *, yes: bool) -> int:
    dataset_dir = dataset_dir.resolve()
    if not _is_lerobot_dataset_dir(dataset_dir):
        print(f"[ERROR] 不是有效的 LeRobot 数据集目录: {dataset_dir}", file=sys.stderr)
        return 2

    if episode_index < 0:
        print("[ERROR] --episode-index 必须 >= 0", file=sys.stderr)
        return 2

    info = _read_json(dataset_dir / "meta" / "info.json")
    episode_infos = _load_episode_infos(dataset_dir)
    if not episode_infos:
        print("[WARN] episodes.jsonl 为空，没有可删除的 episode。")
        return 0

    try:
        _assert_contiguous_episode_infos(episode_infos)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if episode_index not in {e.episode_index for e in episode_infos}:
        print(f"[ERROR] episode_index 不存在: {episode_index}", file=sys.stderr)
        return 2

    for ep in episode_infos:
        parquet_path = _episode_parquet_path(info, dataset_dir, ep.episode_index)
        if not parquet_path.exists():
            print(f"[ERROR] 缺少 parquet 文件: {parquet_path}", file=sys.stderr)
            return 2

    old_total = len(episode_infos)
    moves = _build_moves(episode_infos, episode_index)
    _print_plan(
        dataset_dir=dataset_dir,
        info=info,
        episode_infos=episode_infos,
        delete_index=episode_index,
        moves=moves,
    )

    if not yes:
        print("[DRY-RUN] 未修改任何文件。确认无误后加 --yes 执行硬删除。")
        return 0

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
                info, dataset_dir, src_video, move.new_index
            )
            _move_path(src_video, dst_video)

        for src_audio, dst_audio in _episode_audio_path_map(
            dataset_dir, move.old_index, move.new_index
        ).items():
            _move_path(src_audio, dst_audio)

        _patch_audio_json(
            dataset_dir / "audio" / f"episode_{move.new_index:06d}.audio.json",
            dataset_dir,
            move.new_index,
        )
        _patch_sync_json(
            dataset_dir / "audio" / f"episode_{move.new_index:06d}.sync.json",
            move.new_index,
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

    print("[OK] 已删除指定 episode，并完成后续 episode 重编号。")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="按 episode_index 删除 LeRobot episode，并重排后续编号"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="数据集目录（包含 meta/info.json）。不填则自动在 data/openpi 下找最近修改的数据集。",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="要删除的 episode_index，例如 12。",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="真正执行硬删除。默认不加时只 dry-run 预览。",
    )
    args = parser.parse_args(argv)

    dataset_dir: Optional[Path] = args.dataset
    if dataset_dir is None:
        auto_root = Path("data") / "openpi"
        dataset_dir = _find_latest_dataset_dir(auto_root)
        if dataset_dir is None:
            print(
                f"[ERROR] 未找到数据集目录（在 {auto_root} 下没有 meta/info.json + episodes.jsonl）。",
                file=sys.stderr,
            )
            return 2
        print(f"[AUTO] 选择最近修改的数据集: {dataset_dir}")

    return delete_episode_by_id(
        dataset_dir, int(args.episode_index), yes=bool(args.yes)
    )


if __name__ == "__main__":
    raise SystemExit(main())
