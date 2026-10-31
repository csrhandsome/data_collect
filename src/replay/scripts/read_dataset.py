"""Read the full feature catalog and episode summaries from a local LeRobot dataset."""

from pathlib import Path

from ._common import _episode_summary, _load_metadata, storage_revision
from .read_sidecars import catalog


def read_dataset(root: Path) -> dict:
    """Return JSON-compatible metadata for LeRobot v3.0 MP4 datasets."""
    root = Path(root).resolve()
    info, episodes = _load_metadata(root)
    return {
        "id": root.name,
        "name": info.get("name", root.name),
        "version": info["codebase_version"],
        "storage_revision": storage_revision(root),
        "robot_type": info.get("robot_type"),
        "fps": info["fps"],
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "features": catalog(root, info, next(iter(episodes), 0)),
        "episodes": [_episode_summary(info, episode) for episode in episodes.values()],
    }
