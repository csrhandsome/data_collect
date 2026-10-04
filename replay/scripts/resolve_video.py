"""Locate a camera MP4 and the episode's exclusive timestamp interval inside it."""

from pathlib import Path

from ._common import (
    _V2_VIDEO,
    _V3_VIDEO,
    _finite_number,
    _format_path,
    _get_episode,
    _integer,
    _load_metadata,
)


def resolve_video(root: Path, episode_index: int, feature: str) -> dict:
    """Resolve an MP4 path confined to the dataset root, honoring v3 shard offsets."""
    root = Path(root).resolve()
    info, episodes = _load_metadata(root)
    episode = _get_episode(episodes, episode_index)
    if feature not in info["features"]:
        raise KeyError(f"Feature {feature} not found")
    if info["features"][feature]["dtype"] == "image":
        from .prepare_video import prepare_video

        return prepare_video(root, episode_index, feature)
    if info["features"][feature]["dtype"] != "video":
        raise ValueError("Requested feature is not a video stream")
    feature_info = info["features"][feature].get("info", {})
    if not isinstance(feature_info, dict):
        raise ValueError("Malformed video feature metadata")
    fps = _finite_number(feature_info.get("video.fps", info["fps"]), "video fps", positive=True)
    if info["codebase_version"].startswith("v2."):
        path = _format_path(
            root,
            info.get("video_path", _V2_VIDEO),
            episode_index=episode_index,
            episode_chunk=episode_index // info.get("chunks_size", 1000),
            video_key=feature,
        )
        start, end = 0.0, episode["length"] / info["fps"]
    else:
        prefix = f"videos/{feature}"
        chunk = _integer(episode.get(f"{prefix}/chunk_index"), "video chunk")
        file = _integer(episode.get(f"{prefix}/file_index"), "video file")
        start = _finite_number(episode.get(f"{prefix}/from_timestamp"), "video start")
        end = _finite_number(episode.get(f"{prefix}/to_timestamp"), "video end")
        if start < 0 or end <= start:
            raise ValueError("Invalid episode video timestamp boundaries")
        path = _format_path(
            root,
            info.get("video_path", _V3_VIDEO),
            chunk_index=chunk,
            file_index=file,
            video_key=feature,
        )
    return {
        "path": path,
        "feature": feature,
        "start_time_s": start,
        "end_time_s": end,
        "duration_s": end - start,
        "fps": fps,
    }
