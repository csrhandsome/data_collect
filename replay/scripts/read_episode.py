"""Read a single episode's summary and complete feature catalog."""

from pathlib import Path

from ._common import _episode_summary, _get_episode, _load_metadata
from .read_sidecars import catalog, read_sync


def read_episode(root: Path, episode_index: int) -> dict:
    """Return episode metadata; unknown episodes raise KeyError."""
    info, episodes = _load_metadata(Path(root))
    episode = _get_episode(episodes, episode_index)
    result = {**_episode_summary(info, episode), "blocks": catalog(root, info, episode_index)}
    sync = read_sync(root, episode_index)
    success = sync.get("success")
    if success is not None and not isinstance(success, bool):
        raise ValueError("Episode success must be boolean or null")
    stamp = sync.get("saved_at_ns")
    if stamp is not None and (isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= 0):
        raise ValueError("Invalid episode publication timestamp")
    # Nanosecond epoch values exceed JavaScript's exact integer range.
    result.update(success=success, saved_at_ns=str(stamp) if stamp is not None else None)
    frames = sync.get("frame_records", [])
    if frames:
        if len(frames) != episode["length"]:
            raise ValueError("Sync frame count disagrees with episode")
        result["duration_s"] = (
            frames[-1]["host_frame_monotonic_ns"] - frames[0]["host_frame_monotonic_ns"]
        ) / 1e9 + 1 / info["fps"]
    return result
