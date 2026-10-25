"""WireFishing-M: Harvard Dataverse parts. Layout is filled in once files are on disk."""

from __future__ import annotations

from pathlib import Path

from data_analysis.data_clean.base import BaseAligner
from data_analysis.data_clean.lerobot_v3 import AlignedEpisode


class WireFishingAligner(BaseAligner):
    name = "wirefishing"
    robot_type = "wirefishing"

    def episodes(self, source: Path) -> list[str]:
        files = [path for path in source.rglob("*") if path.is_file()]
        if not files:
            raise FileNotFoundError(f"WireFishing-M is not on disk yet: {source}")
        raise RuntimeError(
            "WireFishing-M files are present, but the episode layout is not mapped yet: "
            + ", ".join(str(path.relative_to(source)) for path in files[:8])
        )

    def load(self, source: Path, episode_id: str, work: Path) -> AlignedEpisode:
        raise RuntimeError(episode_id)
