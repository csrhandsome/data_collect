"""FreeTacMan: task folders of camera1/2/3 mp4 plus a TCP trajectory csv."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from data_analysis.data_clean.base import BaseAligner
from data_analysis.data_clean.lerobot_v3 import GRIPPER_IMAGE_LEFT, GRIPPER_IMAGE_RIGHT, AlignedEpisode

# camera1/camera2 are the two saturated visuo-tactile views. camera3 is the gray wrist RGB.
_CAMERAS = {
    "camera1": GRIPPER_IMAGE_LEFT,
    "camera2": GRIPPER_IMAGE_RIGHT,
    "camera3": "observation.wrist_image_left",
}


class FreeTacManAligner(BaseAligner):
    name = "freetacman"
    robot_type = "freetacman"

    def episodes(self, source: Path) -> list[str]:
        stems = {}
        for path in source.rglob("*_traj.csv"):
            stem = path.name[: -len("_traj.csv")]
            folder = path.parent.relative_to(source)
            stems[f"{folder}/{stem}"] = path
        return sorted(stems)

    def load(self, source: Path, episode_id: str, work: Path) -> AlignedEpisode:
        folder, stem = episode_id.rsplit("/", 1)
        root = source / folder
        videos = {}
        for camera, feature in _CAMERAS.items():
            path = root / f"{stem}_{camera}.mp4"
            if not path.is_file():
                raise FileNotFoundError(path)
            videos[feature] = path
        with (root / f"{stem}_traj.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError(f"{episode_id} trajectory is empty")
        position = np.array(
            [[float(row["TCP_pos_x"]), float(row["TCP_pos_y"]), float(row["TCP_pos_z"])] for row in rows],
            dtype=np.float32,
        )
        euler = np.array(
            [
                [float(row["TCP_euler_x"]), float(row["TCP_euler_y"]), float(row["TCP_euler_z"])]
                for row in rows
            ],
            dtype=np.float32,
        )
        gripper = np.array([[float(row["gripper_distance"])] for row in rows], dtype=np.float32)
        ee_pose = np.concatenate([position, euler], axis=1)
        timestamps = np.array([float(row["timestamp"]) for row in rows], dtype=np.float64)
        gaps = np.diff(timestamps)
        gaps = gaps[gaps > 0]
        fps = float(1.0 / np.median(gaps)) if len(gaps) else 20.0
        columns = {
            "observation.ee_pose": ee_pose,
            "observation.ee_position": position,
            "observation.gripper_position": gripper,
            "action.ee_pose": np.vstack([ee_pose[1:], ee_pose[-1:]]),
            "action.gripper_position": np.vstack([gripper[1:], gripper[-1:]]),
        }
        work.mkdir(parents=True, exist_ok=True)
        return AlignedEpisode(key=episode_id, task=folder, fps=fps, videos=videos, columns=columns)
