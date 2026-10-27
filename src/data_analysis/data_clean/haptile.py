"""HapTile: one zip per task, each episode is trajectory.h5 with MP4 blobs."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from data_analysis.data_clean.base import BaseAligner, SkipEpisode
from data_analysis.data_clean.lerobot_v3 import (
    GRIPPER_IMAGE_LEFT,
    GRIPPER_IMAGE_RIGHT,
    AlignedEpisode,
)

_VIDEOS = {
    "videos/base_camera_rgb_0": "observation.exterior_image",
    "videos/base_camera_rgb_1": "observation.wrist_image_left",
    "videos/tactile_left_rgb": GRIPPER_IMAGE_LEFT,
    "videos/tactile_right_rgb": GRIPPER_IMAGE_RIGHT,
}


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


class HapTileAligner(BaseAligner):
    name = "haptile"
    robot_type = "haptile"

    def episodes(self, source: Path) -> list[str]:
        found = []
        for archive in sorted(source.rglob("*.zip")):
            with zipfile.ZipFile(archive) as handle:
                for name in handle.namelist():
                    if name.endswith("trajectory.h5"):
                        found.append(f"{archive.relative_to(source)}::{name}")
        return found

    def load(self, source: Path, episode_id: str, work: Path) -> AlignedEpisode:
        import h5py

        archive_rel, member = episode_id.split("::", 1)
        work.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source / archive_rel) as handle:
            payload = handle.read(member)
        h5_path = work / "trajectory.h5"
        h5_path.write_bytes(payload)
        videos = {}
        with h5py.File(h5_path, "r") as handle:
            for source_key, feature in _VIDEOS.items():
                blob = bytes(handle[source_key][()])
                path = work / f"{feature.replace('.', '_')}.mp4"
                path.write_bytes(blob)
                videos[feature] = path
            joints = np.asarray(handle["frames/joint_positions"], dtype=np.float32)
            ee_pose = np.asarray(handle["frames/ee_pos_quat"], dtype=np.float32)
            gripper = np.asarray(handle["frames/gripper_position"], dtype=np.float32)
            task = _text(handle["language_instruction"][()])
        if gripper.ndim == 1:
            gripper = gripper.reshape(-1, 1)
        length = min(len(joints), len(ee_pose), len(gripper))
        if length < 2:
            raise SkipEpisode(f"{episode_id} has {length} aligned frames")
        joints = joints[:length]
        ee_pose = ee_pose[:length]
        gripper = gripper[:length]
        columns = {
            "observation.joint_position": joints,
            "observation.ee_pose": ee_pose[:, :6],
            "observation.ee_position": ee_pose[:, :3],
            "observation.gripper_position": gripper[:, :1],
            "action.joint_position": np.vstack([joints[1:], joints[-1:]]),
            "action.ee_pose": np.vstack([ee_pose[1:, :6], ee_pose[-1:, :6]]),
            "action.gripper_position": np.vstack([gripper[1:, :1], gripper[-1:, :1]]),
        }
        return AlignedEpisode(key=episode_id, task=task, fps=15.0, videos=videos, columns=columns)
