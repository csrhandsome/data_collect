import shutil
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _looks_like_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "meta" / "episodes.jsonl").is_file()


def _find_orphan_next_episode_image_dirs(meta) -> list[Path]:
    images_dir = meta.root / "images"
    if not images_dir.is_dir():
        return []

    next_episode = meta.total_episodes
    return sorted(
        {path for path in images_dir.rglob(f"episode_{next_episode:06d}") if path.is_dir()}
    )


def _cleanup_orphan_next_episode_images(meta) -> None:
    leftover_dirs = _find_orphan_next_episode_image_dirs(meta)
    if not leftover_dirs:
        return

    for leftover_dir in leftover_dirs:
        shutil.rmtree(leftover_dir, ignore_errors=True)

    images_dir = meta.root / "images"
    if images_dir.is_dir():
        for child in sorted(images_dir.iterdir()):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
        try:
            images_dir.rmdir()
        except OSError:
            pass

    print(f"[LeRobot] Removed leftover temporary images for episode_{meta.total_episodes:06d}.")


def _resume_existing_dataset_for_recording(repo_id: str, path: Path) -> LeRobotDataset:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.video_utils import get_safe_default_codec

    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=path)

    missing: list[Path] = []
    for ep_idx in range(meta.total_episodes):
        fpath = meta.root / meta.get_data_file_path(ep_idx)
        if not fpath.is_file():
            missing.append(fpath)
    if missing:
        preview = "\n".join(f"  - {p}" for p in missing[:10])
        more = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
        raise RuntimeError(
            "Cannot resume: dataset is missing episode parquet files:\n"
            f"{preview}{more}\n"
            "Fix the dataset first, then retry."
        )

    leftover = _find_orphan_next_episode_image_dirs(meta)
    if leftover:
        _cleanup_orphan_next_episode_images(meta)
        leftover = _find_orphan_next_episode_image_dirs(meta)
        if leftover:
            images_dir = meta.root / "images"
            next_ep = meta.total_episodes
            raise RuntimeError(
                "Cannot resume: found leftover temporary images for the next episode. "
                f"Please remove '{images_dir}' (or the episode_{next_ep:06d} folder) and retry."
            )

    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.meta = meta
    dataset.repo_id = meta.repo_id
    dataset.root = meta.root
    dataset.revision = None
    dataset.tolerance_s = 1e-4
    dataset.image_writer = None
    dataset.episode_buffer = dataset.create_episode_buffer()
    dataset.episodes = None
    dataset.hf_dataset = dataset.create_hf_dataset()
    dataset.image_transforms = None
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    dataset.episode_data_index = None
    dataset.video_backend = get_safe_default_codec()
    dataset.start_image_writer(num_processes=0, num_threads=6)
    return dataset


def _prepare_episode_for_save(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        return
    gripper_values = dataset.episode_buffer.get("gripper_position")
    if not isinstance(gripper_values, list) or not gripper_values:
        return
    dataset.episode_buffer["gripper_position"] = [
        float(v.reshape(-1)[0]) if isinstance(v, np.ndarray) else float(v) for v in gripper_values
    ]


def _discard_unsaved_episode(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        return

    wait_image_writer = getattr(dataset, "_wait_image_writer", None)
    if callable(wait_image_writer):
        wait_image_writer()

    dataset.clear_episode_buffer()
