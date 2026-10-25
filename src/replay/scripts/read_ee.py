"""Extract an episode-isolated EE trace with bounded, endpoint-preserving downsampling."""

from pathlib import Path

import numpy as np

from ._common import (
    _episode_table,
    _feature_blocks,
    _get_episode,
    _integer,
    _load_metadata,
)


def read_ee(
    root: Path, episode_index: int, feature: str = "ee_pose", max_points: int = 2000
) -> dict:
    """Read position/pose vectors without importing LeRobot, Torch, or ROS.

    Positions use meters; 6D pose rotations use radians; 7D quaternion values are
    dimensionless. Bounds describe the complete episode, including skipped rows.
    """
    _integer(max_points, "max_points", minimum=2)
    info, episodes = _load_metadata(Path(root))
    episode = _get_episode(episodes, episode_index)
    blocks = {block["key"]: block for block in _feature_blocks(info)}
    if feature not in blocks and feature == "ee_position":
        from .read_series import read_series

        return read_series(root, episode_index, feature, max_points)
    if feature not in blocks:
        raise KeyError(f"Feature {feature} not found")
    block = blocks[feature]
    if block["kind"] != "ee":
        raise ValueError("Requested feature does not support EE visualization")
    table = _episode_table(Path(root), info, episode, feature)
    try:
        values = np.asarray(table[feature].to_pylist(), dtype=np.float64)
        timestamps = (
            np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
            if "timestamp" in table.column_names
            else np.arange(table.num_rows, dtype=np.float64) / info["fps"]
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("Malformed EE values or timestamps") from exc
    dimension = block["shape"][0]
    if values.shape != (table.num_rows, dimension) or not np.all(np.isfinite(values)):
        raise ValueError("EE values must be finite vectors matching the feature shape")
    if (
        timestamps.shape != (table.num_rows,)
        or not np.all(np.isfinite(timestamps))
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError("Episode timestamps must be finite and strictly increasing")
    timestamps = timestamps - timestamps[0]
    from .read_sidecars import origin_ns, read_sync

    frames = read_sync(root, episode_index).get("frame_records", [])
    if frames:
        if len(frames) != len(values):
            raise ValueError("Sync EE frame count disagrees with episode")
        timestamps = (
            np.asarray([r["host_frame_monotonic_ns"] for r in frames], dtype=np.int64)
            - origin_ns({"frame_records": frames})
        ) / 1e9
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Sync EE timestamps must be increasing")
    defaults = ["x", "y", "z"] + (
        ["roll", "pitch", "yaw"] if dimension == 6 else ["qx", "qy", "qz", "qw"]
    )
    names = block["names"] or defaults[:dimension]
    if len(names) != dimension:
        raise ValueError("EE feature names disagree with its vector dimension")
    units = ["m"] * 3 + (["rad"] * 3 if dimension == 6 else ["1"] * (dimension - 3))
    indices = np.linspace(0, table.num_rows - 1, min(max_points, table.num_rows), dtype=int)
    return {
        "feature": feature,
        "names": names,
        "units": units,
        "timestamps": timestamps[indices].tolist(),
        "values": values[indices].tolist(),
        "point_count": len(indices),
        "total_points": table.num_rows,
        "bounds": {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()},
    }
