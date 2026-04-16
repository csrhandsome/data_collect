import shutil
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _image_feature(image_hw: int) -> dict[str, object]:
    return {
        "dtype": "image",
        "shape": (image_hw, image_hw, 3),
        "names": ["height", "width", "channel"],
    }


def _scalar_feature(name: str, *, dtype: str = "float32") -> dict[str, object]:
    return {
        "dtype": dtype,
        "shape": (1,),
        "names": [name],
    }


def _create_dataset(
    repo_id: str, *, fps: float, image_hw: int, root: Path
) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=float(fps),
        root=root,
        features={
            "exterior_image_1_left": _image_feature(image_hw),
            "exterior_image_2_left": _image_feature(image_hw),
            "wrist_image_left": _image_feature(image_hw),
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=6,
        image_writer_processes=0,
    )


def _create_dataset_force(
    repo_id: str, *, fps: float, image_hw: int, root: Path
) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=float(fps),
        root=root,
        features={
            "exterior_image_1_left": _image_feature(image_hw),
            "exterior_image_2_left": _image_feature(image_hw),
            "wrist_image_left": _image_feature(image_hw),
            "gripper_image_left": _image_feature(image_hw),
            "gripper_image_right": _image_feature(image_hw),
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": _scalar_feature("gripper_position"),
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
            "external_camera_timestamp_ms": _scalar_feature(
                "external_camera_timestamp_ms"
            ),
            "wrist_camera_timestamp_ms": _scalar_feature("wrist_camera_timestamp_ms"),
            "external_camera_frame_age_s": _scalar_feature(
                "external_camera_frame_age_s"
            ),
            "wrist_camera_frame_age_s": _scalar_feature("wrist_camera_frame_age_s"),
        },
        image_writer_threads=6,
        image_writer_processes=0,
    )


def _looks_like_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (
        path / "meta" / "episodes.jsonl"
    ).is_file()


def _find_orphan_next_episode_image_dirs(meta) -> list[Path]:
    images_dir = meta.root / "images"
    if not images_dir.is_dir():
        return []

    next_episode = meta.total_episodes
    return sorted(
        {
            path
            for path in images_dir.rglob(f"episode_{next_episode:06d}")
            if path.is_dir()
        }
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

    print(
        "[LeRobot] Removed leftover temporary images for "
        f"episode_{meta.total_episodes:06d}."
    )


def _resume_existing_dataset_for_recording(
    repo_id: str, path: Path
) -> LeRobotDataset:
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


def _load_or_create_dataset(
    repo_id: str, *, fps: float, image_hw: int, root: Path
) -> LeRobotDataset:
    if root.exists():
        if not root.is_dir():
            raise RuntimeError(f"Dataset path exists and is not a directory: {root}")
        if not _looks_like_lerobot_dataset(root):
            raise RuntimeError(
                "Dataset directory exists but doesn't look like a LeRobot dataset "
                f"(missing meta/info.json): {root}"
            )
        try:
            return _resume_existing_dataset_for_recording(repo_id, root)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load existing LeRobot dataset. "
                "Refusing to modify/recreate automatically. "
                "If the last saved episode is incomplete, prune it manually with: "
                f".venv/bin/python data_analysis/delete_latest_episode.py --dataset {root}. "
                "If the failure mentions leftover temporary images, remove the orphaned "
                f"'{root / 'images'}' episode directory and retry."
            ) from exc

    return _create_dataset(repo_id, fps=fps, image_hw=image_hw, root=root)


def _load_or_create_dataset_force(
    repo_id: str, *, fps: float, image_hw: int, root: Path
) -> LeRobotDataset:
    if root.exists():
        if not root.is_dir():
            raise RuntimeError(f"Dataset path exists and is not a directory: {root}")
        if not _looks_like_lerobot_dataset(root):
            raise RuntimeError(
                "Dataset directory exists but doesn't look like a LeRobot dataset "
                f"(missing meta/info.json): {root}"
            )
        try:
            return _resume_existing_dataset_for_recording(repo_id, root)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load existing LeRobot dataset. "
                "Refusing to modify/recreate automatically. "
                "If the last saved episode is incomplete, prune it manually with: "
                f".venv/bin/python data_analysis/delete_latest_episode.py --dataset {root}. "
                "If the failure mentions leftover temporary images, remove the orphaned "
                f"'{root / 'images'}' episode directory and retry."
            ) from exc

    return _create_dataset_force(repo_id, fps=fps, image_hw=image_hw, root=root)


def _prepare_episode_for_save(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        return
    gripper_values = dataset.episode_buffer.get("gripper_position")
    if not isinstance(gripper_values, list) or not gripper_values:
        return
    dataset.episode_buffer["gripper_position"] = [
        float(v.reshape(-1)[0]) if isinstance(v, np.ndarray) else float(v)
        for v in gripper_values
    ]


def _prepare_episode_for_save_force(dataset: LeRobotDataset) -> None:
    _prepare_episode_for_save(dataset)


def _discard_unsaved_episode(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        return

    wait_image_writer = getattr(dataset, "_wait_image_writer", None)
    if callable(wait_image_writer):
        wait_image_writer()

    dataset.clear_episode_buffer()
