"""Shared trial alignment: map one source dataset onto LeRobot v3."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from data_analysis.data_clean.lerobot_v3 import AlignedEpisode, DatasetConverter, write_dataset


class SkipEpisode(Exception):
    """One source episode cannot be aligned. The rest of the dataset continues."""


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

    def iter_episodes(self, source: Path, work: Path, done: set[str] | None = None):
        finished = done or set()
        for index, episode_id in enumerate(self.episodes(source)):
            if episode_id in finished:
                continue
            folder = work / f"ep-{index:06d}"
            if folder.exists():
                shutil.rmtree(folder)
            try:
                episode = self.load(source, episode_id, folder)
            except SkipEpisode as exc:
                print(f"[{self.name}] skip {episode_id}: {exc}", flush=True)
                shutil.rmtree(folder, ignore_errors=True)
                continue
            yield episode
            shutil.rmtree(folder, ignore_errors=True)

    def convert_all(self, source: Path, output: Path) -> Path:
        source = Path(source)
        output = Path(output)
        if not source.exists():
            raise FileNotFoundError(source)
        work = output.parent / f".{self.name}-work"
        work.mkdir(parents=True, exist_ok=True)
        writer = DatasetConverter(output, self.robot_type)
        for episode in self.iter_episodes(source, work, writer.done_ids()):
            writer.add(episode)
            print(f"[{self.name}] {writer.state['episodes'][-1]['episode_index']} {episode.key}", flush=True)
        if not writer.state["episodes"]:
            raise RuntimeError(f"{self.name} produced no episodes from {source}")
        return writer.close()
