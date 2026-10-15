"""Private transactional deletion engine; callers use EpisodeDeleter."""

from __future__ import annotations

import ctypes
import fcntl
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa

from control.collection.deletion import EpisodeDeletionResult

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    length: int


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _is_lerobot_dataset_dir(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


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


def _assert_contiguous_episode_infos(episode_infos: List[EpisodeInfo]) -> None:
    expected = list(range(len(episode_infos)))
    actual = [e.episode_index for e in episode_infos]
    if actual != expected:
        raise RuntimeError(
            "episode 元数据里的 episode_index 不是连续的 0..N-1，拒绝自动重排。\n"
            f"Actual: {actual[:20]}{'...' if len(actual) > 20 else ''}"
        )


def _move_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise RuntimeError(f"目标路径已存在，拒绝覆盖: {dst}")
    shutil.move(str(src), str(dst))


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


def _prepare_episode(
    dataset_dir: Path,
    episode_index: int,
    *,
    dry_run: bool,
    final_dataset_dir: Path | None = None,
    output_dir: Path | None = None,
) -> EpisodeDeletionResult:
    from control.collection._v3_deletion import prepare_episode
    from replay.scripts._common import _load_metadata

    dataset_dir = dataset_dir.resolve()
    if not _is_lerobot_dataset_dir(dataset_dir):
        raise FileNotFoundError(f"不是有效的 LeRobot 数据集目录: {dataset_dir}")
    _load_metadata(dataset_dir)
    if episode_index < 0:
        raise ValueError("episode_index must be >= 0")
    return prepare_episode(
        dataset_dir, episode_index, dry_run, final_dataset_dir or dataset_dir, output_dir=output_dir
    )


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
    from data_analysis.dataset_io import episode_dataframe, episode_paths, episode_rows

    episodes = [EpisodeInfo(row["episode_index"], row["length"]) for row in episode_rows(root)]
    paths = episode_paths(root)
    _assert_contiguous_episode_infos(episodes)
    if info["total_episodes"] != len(episodes) or info["total_frames"] != sum(
        ep.length for ep in episodes
    ):
        raise ValueError("删除后的元数据总数不一致")
    start = 0
    for episode in episodes:
        table = pa.Table.from_pandas(
            episode_dataframe(paths[episode.episode_index], episode.episode_index)
        )
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
            logger.info("[STAGING] 正在生成删除后的 v3 MP4 数据集。")
            result = _prepare_episode(
                dataset_dir,
                episode_index,
                dry_run=False,
                final_dataset_dir=dataset_dir,
                output_dir=staged,
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
