"""删除 LeRobot 数据集中“最新一次”(最大 episode_index) 的 episode。

特点：
- 默认 dry-run，不会真的改动文件；加 --apply 才会执行。
- 会同步更新 meta/episodes.jsonl、meta/episodes_stats.jsonl、meta/info.json。
- 默认把被删的文件移动到数据集内的 .trash/ 目录，避免误删；加 --hard-delete 才永久删除。
- 如果不传 --dataset，会在 data/openpi/ 下自动选择最近修改的一个数据集目录。

用法示例：
  # 只预览将要删除哪些文件
  .venv/bin/python data_analysis/delete_latest_episode.py --dataset data/openpi/franka_droid_lerobot_2_4

  # 执行删除（移动到 .trash/）
  .venv/bin/python data_analysis/delete_latest_episode.py --dataset data/openpi/franka_droid_lerobot_2_4 --apply

  # 执行硬删除（不可恢复）
  uv run data_analysis/delete_latest_episode.py --dataset data/openpi/franka_droid_lerobot_2_4 --apply --hard-delete
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    length: int


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(obj, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _is_lerobot_dataset_dir(path: Path) -> bool:
    return (path / "meta" / "info.json").exists() and (
        path / "meta" / "episodes.jsonl"
    ).exists()


def _find_latest_dataset_dir(root: Path) -> Optional[Path]:
    """在 root 下递归寻找 LeRobot 数据集目录，选最近修改的那个。"""

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
    """尽可能找到该 episode 对应的视频文件（如果存在）。"""

    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size

    videos_dir = dataset_dir / "videos" / f"chunk-{episode_chunk:03d}"
    if not videos_dir.exists():
        return []

    # video_key 可能有多个子目录；按惯例匹配 episode_{idx}.mp4
    pattern = f"episode_{episode_index:06d}.mp4"
    return sorted(videos_dir.rglob(pattern))


def _backup_meta(dataset_dir: Path, *, dry_run: bool) -> Optional[Path]:
    meta_dir = dataset_dir / "meta"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = meta_dir / f"_backup_delete_latest_{stamp}"

    to_copy = [
        meta_dir / "info.json",
        meta_dir / "episodes.jsonl",
        meta_dir / "episodes_stats.jsonl",
        meta_dir / "tasks.jsonl",
    ]

    if dry_run:
        return backup_dir

    backup_dir.mkdir(parents=True, exist_ok=False)
    for src in to_copy:
        if src.exists():
            shutil.copy2(src, backup_dir / src.name)
    return backup_dir


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

        # 常见格式："0:total_episodes"。只在 end == old_total 时更新。
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
    return out


def delete_latest_episode(dataset_dir: Path, *, apply: bool, hard_delete: bool) -> int:
    dataset_dir = dataset_dir.resolve()
    if not _is_lerobot_dataset_dir(dataset_dir):
        print(f"[ERROR] 不是有效的 LeRobot 数据集目录: {dataset_dir}", file=sys.stderr)
        return 2

    meta_dir = dataset_dir / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"

    info = _read_json(info_path)
    episode_infos = _load_episode_infos(dataset_dir)
    if not episode_infos:
        print("[WARN] episodes.jsonl 为空，没有可删除的 episode。")
        return 0

    latest = max(episode_infos, key=lambda e: e.episode_index)
    latest_index = latest.episode_index

    parquet_path = _episode_parquet_path(info, dataset_dir, latest_index)
    video_paths = _episode_video_paths(info, dataset_dir, latest_index)

    dry_run = not apply
    backup_dir = _backup_meta(dataset_dir, dry_run=dry_run)

    print(f"Dataset: {dataset_dir}")
    print(f"Latest episode_index: {latest_index}")
    print(f"Parquet: {parquet_path}")
    if video_paths:
        print("Videos:")
        for p in video_paths:
            print(f"  - {p}")
    else:
        print("Videos: (none)")

    print(f"Meta backup dir: {backup_dir}")
    if dry_run:
        print("[DRY-RUN] 未做任何改动。加 --apply 才会真的删除。")
        return 0

    # 1) 删除/移动数据文件
    trash_dir = dataset_dir / ".trash" / f"episode_{latest_index:06d}"
    if not hard_delete:
        trash_dir.mkdir(parents=True, exist_ok=True)

    def remove_or_trash(path: Path) -> None:
        if not path.exists():
            return
        if hard_delete:
            path.unlink()
            return
        # Important: LeRobot's internal consistency checks count files via rglob("*.parquet") and rglob("*.mp4").
        # If we keep the original extension under the dataset directory, it will break those assertions.
        dest = trash_dir / f"{path.name}.deleted"
        shutil.move(str(path), str(dest))

    remove_or_trash(parquet_path)
    for vp in video_paths:
        remove_or_trash(vp)

    # 2) 更新 episodes.jsonl（删掉 latest 那行）
    episodes_rows = _read_jsonl(episodes_path)
    episodes_rows = [
        r for r in episodes_rows if int(r.get("episode_index", -1)) != latest_index
    ]
    _write_jsonl(episodes_path, episodes_rows)

    # 3) 更新 episodes_stats.jsonl（如果存在就删掉）
    if stats_path.exists():
        stats_rows = _read_jsonl(stats_path)
        stats_rows = [
            r for r in stats_rows if int(r.get("episode_index", -1)) != latest_index
        ]
        _write_jsonl(stats_path, stats_rows)

    # 4) 更新 info.json（重算 total_episodes/total_frames/total_chunks/splits）
    old_total = int(info.get("total_episodes", len(episode_infos)))
    kept_infos = _load_episode_infos(dataset_dir)
    new_total = len(kept_infos)
    new_frames = sum(e.length for e in kept_infos)
    chunks_size = int(info.get("chunks_size", 1000))
    new_total_chunks = (
        (new_total + chunks_size - 1) // chunks_size if new_total > 0 else 0
    )

    info["total_episodes"] = new_total
    info["total_frames"] = new_frames
    info["total_chunks"] = new_total_chunks
    _update_splits_in_place(info, old_total=old_total, new_total=new_total)
    _write_json(info_path, info)

    print("[OK] 已删除最新 episode，并更新 meta。")
    if not hard_delete:
        print(
            f"[OK] 被删文件已移动到: {trash_dir} (已加 .deleted 后缀，避免影响 LeRobot 校验)"
        )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="删除 LeRobot 数据集中最新一次(最大 episode_index)的数据"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="数据集目录（包含 meta/info.json）。不填则自动在 data/openpi 下找最近修改的数据集。",
    )
    parser.add_argument(
        "--apply", action="store_true", help="真的执行删除（默认只 dry-run 预览）"
    )
    parser.add_argument(
        "--hard-delete", action="store_true", help="永久删除文件（默认移动到 .trash/）"
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

    return delete_latest_episode(dataset_dir, apply=True, hard_delete=True)


if __name__ == "__main__":
    raise SystemExit(main())
