"""Prepare v3 deletion inside the existing locked directory transaction."""

import shutil
from pathlib import Path

from control.collection.deletion import EpisodeDeletionResult


def prepare_episode(
    root: Path, index: int, dry_run: bool, final_root: Path, *, output_dir: Path | None = None
):
    from control.collection._episode_deletion import (
        _episode_audio_path_map,
        _episode_audio_paths,
        _move_path,
        _patch_audio_json,
        _patch_sync_json,
        _source_manifest,
    )
    from data_analysis.dataset_io import episode_rows

    _source_manifest(root)
    rows = episode_rows(root)
    if [row["episode_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("Episode indices must be contiguous before deletion")
    if index >= len(rows):
        raise ValueError(f"episode_index 不存在: {index}")
    total = sum(row["length"] for row in rows)
    result = EpisodeDeletionResult(
        final_root,
        index,
        rows[index]["length"],
        len(rows),
        len(rows) - 1,
        total,
        total - rows[index]["length"],
        dry_run,
    )
    if dry_run:
        return result

    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import delete_episodes

    if output_dir is None:
        raise ValueError("v3 deletion requires a separate output directory")
    output = output_dir.resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Deletion output must be separate from the source dataset")
    dataset = LeRobotDataset(root.name, root=root, video_backend="pyav")
    if len(rows) == 1:
        # Upstream refuses deleting every episode; retain a resumable empty schema.
        edited = LeRobotDataset.create(
            root.name,
            dataset.fps,
            dataset.features,
            root=output,
            robot_type=dataset.meta.robot_type,
            use_videos=bool(dataset.meta.video_keys),
        )
    else:
        edited = delete_episodes(dataset, [index], output_dir=output, repo_id=root.name)
    edited.finalize()

    # Upstream generates data/videos/meta directly from the read-only source.
    # Copy only auxiliary files, excluding the deleted episode's sidecars.
    deleted_paths = set(_episode_audio_paths(root, index))

    def ignore_deleted(directory, names):
        return [name for name in names if Path(directory) / name in deleted_paths]

    for path in root.iterdir():
        if path.name in {"data", "videos", "meta"} or path in deleted_paths:
            continue
        target = output / path.name
        if path.is_dir():
            shutil.copytree(path, target, ignore=ignore_deleted)
        else:
            shutil.copy2(path, target)
    for old in range(len(rows)):
        if old == index:
            continue
        new = old - int(old > index)
        if old != new:
            for source, target in _episode_audio_path_map(output, old, new).items():
                _move_path(source, target)
        _patch_audio_json(output / "audio" / f"episode_{new:06d}.audio.json", final_root, new)
        for folder in ("audio", ""):
            _patch_sync_json(output / folder / f"episode_{new:06d}.sync.json", new, old, final_root)
    return result
