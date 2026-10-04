"""Local recording lifecycle for LeRobot 0.6.1 / dataset v3.0."""

import json
import shutil
from pathlib import Path

from control.util.hub_compat import allow_hub1_for_transformers

allow_hub1_for_transformers()
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def _looks_like_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _find_orphan_next_episode_image_dirs(meta) -> list[Path]:
    images_dir = meta.root / "images"
    if not images_dir.is_dir():
        return []
    return sorted(
        path for path in images_dir.rglob(f"episode_{meta.total_episodes:06d}") if path.is_dir()
    )


def _cleanup_orphan_next_episode_images(meta) -> None:
    for path in _find_orphan_next_episode_image_dirs(meta):
        shutil.rmtree(path)


def _ensure_local_recording_metadata(path: Path) -> None:
    from replay.scripts._common import _episode_data_path, _load_metadata

    info = json.loads((path / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            "Recording requires a v3.0 dataset. Select a new dataset.date, or convert a copy "
            "with: uv run python -m scripts.migrate_lerobot_dataset --input PATH --output NEW_PATH"
        )
    required = []
    if info.get("total_tasks", 0):
        required.append(path / "meta/tasks.parquet")
    if info.get("total_episodes", 0) or info.get("total_frames", 0):
        required.append(path / "meta/stats.json")
        if not list((path / "meta/episodes").glob("chunk-*/file-*.parquet")):
            required.append(path / "meta/episodes")
    missing = [item for item in required if not item.exists()]
    if missing:
        raise RuntimeError(f"Cannot resume: dataset is missing local metadata: {missing}")
    try:
        info, episodes = _load_metadata(path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot resume: invalid local metadata: {exc}") from exc
    for episode in episodes.values():
        data_path = _episode_data_path(path, info, episode)
        if not data_path.is_file():
            raise RuntimeError(
                f"Cannot resume: dataset is missing episode parquet files: {data_path}"
            )


def _resume_existing_dataset_for_recording(repo_id: str, path: Path) -> LeRobotDataset:
    from control.util.lerobot_metadata import normalize_episode_metadata

    # Check locally first: incomplete metadata must never trigger Hub downloads.
    _ensure_local_recording_metadata(path)
    normalize_episode_metadata(path)
    dataset = LeRobotDataset.resume(
        repo_id=repo_id, root=path, image_writer_processes=0, image_writer_threads=4
    )
    try:
        _cleanup_orphan_next_episode_images(dataset.meta)
    except BaseException:
        dataset.finalize()
        raise
    return dataset


def _discard_unsaved_episode(dataset: LeRobotDataset) -> None:
    # The image writer must finish before temporary images can be removed.
    writer = dataset.writer
    if writer is not None:
        if writer.image_writer is not None:
            writer.image_writer.wait_until_done()
        dataset.clear_episode_buffer()
