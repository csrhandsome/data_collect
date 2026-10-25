"""Shared trial alignment: map one source dataset onto LeRobot v3."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from data_analysis.data_clean.lerobot_v3 import AlignedEpisode, write_dataset


class BaseAligner(ABC):
    name: str
    robot_type: str = "unknown"

    @abstractmethod
    def episodes(self, source: Path) -> list[str]:
        """Stable episode ids. The trial uses only the first `limit` of these."""

    @abstractmethod
    def load(self, source: Path, episode_id: str, work: Path) -> AlignedEpisode:
        """Materialize one episode under `work` and return v3 columns plus MP4 paths."""

    def align(self, source: Path, output: Path, *, limit: int = 1) -> Path:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(source)
        if limit < 1:
            raise ValueError("limit must be at least 1")
        ids = self.episodes(source)
        if not ids:
            raise RuntimeError(f"{self.name} has no episodes in {source}")
        chosen = ids[:limit]
        work = Path(output).parent / f".{self.name}-work"
        work.mkdir(parents=True, exist_ok=True)
        loaded = [self.load(source, episode_id, work / episode_id) for episode_id in chosen]
        fps = loaded[0].fps
        return write_dataset(loaded, Path(output), robot_type=self.robot_type, fps=fps)
