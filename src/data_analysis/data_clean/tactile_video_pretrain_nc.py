"""yxma/tactile-video-pretrain-nc episode directories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from data_analysis.data_clean.base import BaseAligner
from data_analysis.data_clean.lerobot_v3 import GRIPPER_IMAGE_LEFT, GRIPPER_IMAGE_RIGHT, AlignedEpisode

# One gripper has two tactile cameras. The other arm's pair is not used.
_VIDEOS = {
    "cam_head.mp4": "observation.exterior_image",
    "cam_hand_left.mp4": "observation.wrist_image_left",
    "tactile_left_l.mp4": GRIPPER_IMAGE_LEFT,
    "tactile_left_r.mp4": GRIPPER_IMAGE_RIGHT,
}


class TactileVideoPretrainAligner(BaseAligner):
    name = "tactile_video_pretrain_nc"
    robot_type = "vtouch"

    def episodes(self, source: Path) -> list[str]:
        found = []
        for path in sorted(source.rglob("joints.parquet")):
            folder = path.parent
            if (folder / "cam_head.mp4").is_file():
                found.append(str(folder.relative_to(source)))
        return found

    def load(self, source: Path, episode_id: str, work: Path) -> AlignedEpisode:
        folder = source / episode_id
        videos = {}
        for filename, feature in _VIDEOS.items():
            path = folder / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            videos[feature] = path
        table = pq.read_table(folder / "joints.parquet")
        columns = _columns_from_joints(table)
        names = table.schema.names
        length = table.num_rows
        if "timestamp" in names:
            stamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
            gaps = np.diff(stamps)
            gaps = gaps[gaps > 0]
            fps = float(1.0 / np.median(gaps)) if len(gaps) else 30.0
        else:
            fps = 30.0
        if not columns:
            zeros = np.zeros((length, 1), dtype=np.float32)
            columns = {"observation.gripper_position": zeros, "action.gripper_position": zeros}
        work.mkdir(parents=True, exist_ok=True)
        return AlignedEpisode(
            key=episode_id,
            task=episode_id.split("/")[0],
            fps=fps,
            videos=videos,
            columns=columns,
        )


def _columns_from_joints(table) -> dict[str, np.ndarray]:
    names = set(table.schema.names)
    columns = {}
    mapping = {
        "observation.state": "observation.joint_position",
        "action": "action.joint_position",
        "observation.ee_pose": "observation.ee_pose",
        "action.ee_pose": "action.ee_pose",
    }
    for source_name, feature in mapping.items():
        if source_name not in names:
            continue
        values = np.asarray(table[source_name].to_pylist(), dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        width = 7 if feature.endswith("joint_position") else 6
        if values.shape[1] < width:
            pad = np.zeros((values.shape[0], width - values.shape[1]), dtype=np.float32)
            values = np.concatenate([values, pad], axis=1)
        columns[feature] = values[:, :width]
    return columns
