"""Touch and Go: each clip folder has video.mp4 and gelsight.mp4."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_analysis.data_clean.base import BaseAligner
from data_analysis.data_clean.lerobot_v3 import GRIPPER_IMAGE_LEFT, AlignedEpisode


class TouchAndGoAligner(BaseAligner):
    name = "touch_and_go"
    robot_type = "touch_and_go"

    def episodes(self, source: Path) -> list[str]:
        found = []
        for tactile in sorted(source.rglob("gelsight.mp4")):
            if (tactile.parent / "video.mp4").is_file():
                found.append(str(tactile.parent.relative_to(source)))
        return found

    def load(self, source: Path, episode_id: str, work: Path) -> AlignedEpisode:
        folder = source / episode_id
        length = 1
        zeros = np.zeros((length, 1), dtype=np.float32)
        work.mkdir(parents=True, exist_ok=True)
        return AlignedEpisode(
            key=episode_id,
            task=episode_id,
            fps=27.0,
            videos={
                "observation.exterior_image": folder / "video.mp4",
                # Only one GelSight stream exists, so the right gripper slot stays empty.
                GRIPPER_IMAGE_LEFT: folder / "gelsight.mp4",
            },
            columns={"observation.gripper_position": zeros, "action.gripper_position": zeros},
        )
