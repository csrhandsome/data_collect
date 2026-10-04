"""Read a single episode's summary and complete feature catalog."""

from pathlib import Path

from ._common import _episode_summary, _get_episode, _load_metadata
from .read_sidecars import catalog, read_sync


def read_episode(root: Path, episode_index: int) -> dict:
    """Return episode metadata; unknown episodes raise KeyError."""
    info, episodes = _load_metadata(Path(root))
    episode = _get_episode(episodes, episode_index)
    result = {**_episode_summary(info, episode), "blocks": catalog(root, info, episode_index)}
    frames = read_sync(root, episode_index).get("frame_records", [])
    if frames:
        if len(frames) != episode["length"]:
            raise ValueError("Sync frame count disagrees with episode")
        result["duration_s"] = (
            frames[-1]["host_frame_monotonic_ns"] - frames[0]["host_frame_monotonic_ns"]
        ) / 1e9 + 1 / info["fps"]
    return result
