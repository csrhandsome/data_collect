"""Decode one embedded PNG/JPEG or dataset-confined image path from Parquet."""

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from ._common import _episode_table, _get_episode, _integer, _load_metadata
from .read_sidecars import confined


def read_image(root, episode_index, feature, frame_index):
    _integer(frame_index, "frame_index", minimum=0)
    info, episodes = _load_metadata(Path(root))
    episode = _get_episode(episodes, episode_index)
    if feature not in info["features"]:
        raise KeyError(feature)
    if info["features"][feature]["dtype"] != "image":
        raise ValueError("Feature is not an embedded image stream")
    if frame_index >= episode["length"]:
        raise KeyError("Image frame missing")
    table = _image_table(
        str(Path(root).resolve()),
        episode_index,
        feature,
        (Path(root) / "meta/info.json").stat().st_mtime_ns,
    )
    value = table[feature][frame_index].as_py()
    if isinstance(value, dict):
        data = value.get("bytes")
        if data is None and value.get("path"):
            data = confined(root, value["path"]).read_bytes()
    else:
        data = value
    if not isinstance(data, bytes):
        raise ValueError("Embedded image bytes are missing")
    with Image.open(io.BytesIO(data)) as image:
        output = io.BytesIO()
        image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


@lru_cache(maxsize=8)
def _image_table(root, episode_index, feature, metadata_stamp):
    info, episodes = _load_metadata(Path(root))
    return _episode_table(Path(root), info, _get_episode(episodes, episode_index), feature)


def read_image_metadata(root, episode_index, feature):
    from .read_sidecars import origin_ns, read_sync

    info, episodes = _load_metadata(Path(root))
    episode = _get_episode(episodes, episode_index)
    if feature not in info["features"] or info["features"][feature]["dtype"] != "image":
        raise ValueError("Feature is not an image stream")
    sync = read_sync(root, episode_index)
    frames = sync.get("frame_records", [])
    timestamps = (
        [(r["host_frame_monotonic_ns"] - origin_ns(sync)) / 1e9 for r in frames]
        if len(frames) == episode["length"]
        else [i / info["fps"] for i in range(episode["length"])]
    )
    if frames and len(frames) != episode["length"]:
        raise ValueError("Sync image frame count disagrees with episode")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Image timestamps must be finite and increasing")
    return {"length": episode["length"], "timestamps": timestamps}
