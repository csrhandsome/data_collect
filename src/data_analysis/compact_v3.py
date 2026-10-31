"""Compact sealed collection shards without changing episode IDs or sidecars."""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from control.collection._episode_deletion import (
    _exchange_directories,
    _source_manifest,
    _sync_directory,
    _validate_result,
)
from data_analysis.merge_v3 import repair_metadata_shard_indices
from replay.scripts._common import _episode_data_path, _format_path, _load_metadata
from replay.scripts.dataset_lock import dataset_access

logger = logging.getLogger(__name__)
_CORE = {"data", "videos", "meta"}


def _core_manifest(manifest):
    return {name: value for name, value in manifest.items() if Path(name).parts[0] in _CORE}


def _needs_compaction(root, info, episodes):
    groups = [(set(_episode_data_path(root, info, ep) for ep in episodes.values()),
               info.get("data_files_size_in_mb", 100))]
    for key, feature in info["features"].items():
        if feature["dtype"] != "video":
            continue
        paths = {
            _format_path(root, info["video_path"], video_key=key,
                         chunk_index=ep[f"videos/{key}/chunk_index"],
                         file_index=ep[f"videos/{key}/file_index"])
            for ep in episodes.values()
        }
        groups.append((paths, info.get("video_files_size_in_mb", 200)))
    groups.append((set((root / "meta/episodes").glob("chunk-*/file-*.parquet")),
                   info.get("data_files_size_in_mb", 100)))
    for paths, limit in groups:
        sizes = [p.stat().st_size for p in sorted(paths)]
        if any(a + b < limit * 1024 * 1024 for a, b in zip(sizes, sizes[1:])):
            return True
    return False


def _prepare(root: Path, staged: Path, info: dict, episodes: dict) -> None:
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.video_utils import get_video_duration_in_s

    # The upstream remuxer copies compressed packets, retaining codec/keyframes.
    aggregate_datasets(
        [root.name], root.name, roots=[root], aggr_root=staged,
        data_files_size_in_mb=info.get("data_files_size_in_mb", 100),
        video_files_size_in_mb=info.get("video_files_size_in_mb", 200),
        chunk_size=info.get("chunks_size", 1000),
    )
    repair_metadata_shard_indices(staged)
    merged_info, merged = _load_metadata(staged)
    # Aggregation's typed info model omits project-specific descriptive fields.
    # Retain these without restoring the source's storage path templates.
    for key, value in info.items():
        if key not in merged_info and key not in {"episodes_path", "storage_format"}:
            merged_info[key] = value
    (staged / "meta/info.json").write_text(json.dumps(merged_info, indent=2) + "\n")
    if merged_info["total_frames"] != info["total_frames"] or list(merged) != list(episodes):
        raise ValueError("Compaction changed episode or frame counts")
    durations = {}
    for index, episode in merged.items():
        for key in ("length", "tasks", "dataset_from_index", "dataset_to_index"):
            if episode.get(key) != episodes[index].get(key):
                raise ValueError(f"Compaction changed episode {index}: {key}")
        for key, feature in merged_info["features"].items():
            if feature["dtype"] != "video":
                continue
            prefix = f"videos/{key}"
            path = _format_path(staged, merged_info["video_path"], video_key=key,
                                chunk_index=episode[f"{prefix}/chunk_index"],
                                file_index=episode[f"{prefix}/file_index"])
            if path not in durations:
                durations[path] = get_video_duration_in_s(path)
            start, end = episode[f"{prefix}/from_timestamp"], episode[f"{prefix}/to_timestamp"]
            tolerance = 1 / info["fps"] + 1e-6
            if (start < 0 or end <= start or end > durations[path] + tolerance
                    or abs(end - start - episode["length"] / info["fps"]) > tolerance):
                raise ValueError(f"Invalid compacted video interval for episode {index}")


def _copy_auxiliary(root: Path, staged: Path, manifest: dict, previous: dict | None = None) -> list[Path]:
    # IDs and dataset root stay fixed: annotations, audio/VAD, timestamps and
    # high-rate traces can be copied byte-for-byte, including unknown sidecars.
    changed = []
    if previous is None:
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if relative.parts[0] not in _CORE and path.is_dir():
                (staged / relative).mkdir(parents=True, exist_ok=True)
    for name, signature in manifest.items():
        if Path(name).parts[0] in _CORE:
            continue
        if previous is not None and previous.get(name) == signature:
            continue
        target = staged / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
        changed.append(target)
    if previous is not None:
        for name in previous.keys() - manifest.keys():
            if Path(name).parts[0] not in _CORE:
                (staged / name).unlink(missing_ok=True)
    return changed


def compact_dataset(root: Path) -> bool:
    """Pack a closed dataset in place; keep original bytes on any prepare failure.

    Call only after the recording writer closes. Readers and annotations remain
    available during the expensive remux, with a short lock for publication.
    """
    root = Path(root).resolve()
    with dataset_access(root):
        info, episodes = _load_metadata(root)
        if len(episodes) < 2 or not _needs_compaction(root, info, episodes):
            return False
        manifest = _source_manifest(root)
        transaction = Path(tempfile.mkdtemp(prefix=f".{root.name}.compact-", dir=root.parent))
        try:
            staged = transaction / "dataset"
            logger.info("正在合并采集文件：%s（%s 个 episode）", root.name, len(episodes))
            _prepare(root, staged, info, episodes)
            _copy_auxiliary(root, staged, manifest)
            _validate_result(staged, root)
            # Flush media before taking the publication lock, so playback is
            # never held up by the bulk of the disk work.
            for path in staged.rglob("*"):
                if path.is_file():
                    with path.open("rb") as file:
                        os.fsync(file.fileno())
            for path in sorted(staged.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if path.is_dir():
                    _sync_directory(path)
            _sync_directory(staged)
            _sync_directory(transaction)
        except BaseException:
            shutil.rmtree(transaction)
            raise
    try:
        with dataset_access(root, exclusive=True):
            current = _source_manifest(root)
            if _core_manifest(current) != _core_manifest(manifest):
                raise RuntimeError("准备期间原数据集被其他进程修改，已取消合并")
            # Include annotations saved while remuxing, under the same gate as
            # update_sync(). Never overwrite a newer human label with a snapshot.
            changed = _copy_auxiliary(root, staged, current, previous=manifest)
            for path in changed:
                with path.open("rb") as file:
                    os.fsync(file.fileno())
            directories = {staged}
            for path in changed:
                directories.update(parent for parent in path.parents if parent.is_relative_to(staged))
            for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
                _sync_directory(path)
            _sync_directory(staged)
            _sync_directory(transaction)
            _exchange_directories(root, staged)
            try:
                _sync_directory(root.parent)
                _sync_directory(transaction)
            except OSError as exc:
                logger.warning("合并已提交，但目录落盘确认失败：%s", exc)
        logger.info("采集文件合并完成：%s；episode 编号及附加记录保持不变", root.name)
    finally:
        shutil.rmtree(transaction)
    return True
