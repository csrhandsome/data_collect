"""Prepare v3 deletion inside the existing locked directory transaction."""

import shutil
import tempfile
from pathlib import Path

from control.collection.deletion import EpisodeDeletionResult


def prepare_episode(root: Path, index: int, dry_run: bool, final_root: Path):
    from control.collection._episode_deletion import (
        _episode_audio_path_map,
        _episode_audio_paths,
        _move_path,
        _patch_audio_json,
        _patch_sync_json,
        _remove_path,
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

    dataset = LeRobotDataset(root.name, root=root, video_backend="pyav")
    with tempfile.TemporaryDirectory(prefix=".v3-delete-", dir=root.parent) as temporary:
        output = Path(temporary) / "dataset"
        if len(rows) == 1:
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
        for name in ("data", "videos", "meta"):
            _remove_path(root / name)
            if (output / name).exists():
                shutil.move(str(output / name), str(root / name))
    for path in _episode_audio_paths(root, index):
        _remove_path(path)
    for old in range(len(rows)):
        if old == index:
            continue
        new = old - int(old > index)
        if old != new:
            for source, target in _episode_audio_path_map(root, old, new).items():
                _move_path(source, target)
        _patch_audio_json(root / "audio" / f"episode_{new:06d}.audio.json", final_root, new)
        for folder in ("audio", ""):
            _patch_sync_json(root / folder / f"episode_{new:06d}.sync.json", new, old, final_root)
    return result
