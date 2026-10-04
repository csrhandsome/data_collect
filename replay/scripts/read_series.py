"""Episode-isolated numeric columns, EE sidecars and high-rate traces."""

import json
from pathlib import Path

import numpy as np

from ._common import _episode_table, _get_episode, _integer, _load_metadata
from .read_sidecars import confined, origin_ns, read_sync


def read_series(root, episode_index, feature, max_points=2000):
    _integer(max_points, "max_points", minimum=2)
    root = Path(root)
    info, episodes = _load_metadata(root)
    episode = _get_episode(episodes, episode_index)
    sync = read_sync(root, episode_index)
    frames = sync.get("frame_records", [])
    if feature.startswith("trace_"):
        selected = feature.removeprefix("trace_")
        if selected not in {"joint_position", "ee_position", "vr_position", "target_ee_position"}:
            raise KeyError(feature)
        path = sync.get("action_trace")
        if not path:
            raise KeyError("Action trace missing")
        with confined(root, path).open() as file:
            records = [json.loads(line) for line in file]
        values = np.asarray([r[selected] for r in records], dtype=np.float64)
        times = (
            np.asarray([r["host_sample_monotonic_ns"] for r in records], dtype=np.int64)
            - origin_ns(sync)
        ) / 1e9
    elif feature in info["features"]:
        table = _episode_table(root, info, episode, feature)
        values = np.asarray(table[feature].to_pylist(), dtype=np.float64)
        times = np.asarray(table["timestamp"].to_pylist(), dtype=float)
        times -= times[0]
        if frames and len(frames) == len(values):
            times = (
                np.asarray([r["host_frame_monotonic_ns"] for r in frames], dtype=np.int64)
                - origin_ns(sync)
            ) / 1e9
    elif frames and feature in frames[0]:
        if len(frames) != episode["length"]:
            raise ValueError("Sidecar frame count disagrees with episode")
        values = np.asarray([r[feature] for r in frames], dtype=np.float64)
        times = (
            np.asarray([r["host_frame_monotonic_ns"] for r in frames], dtype=np.int64)
            - origin_ns(sync)
        ) / 1e9
    else:
        raise KeyError(feature)
    if values.ndim == 1:
        values = values[:, None]
    if (
        not len(values)
        or values.ndim != 2
        or not np.isfinite(values).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Numeric trace must contain finite vectors and increasing timestamps")
    names = info["features"].get(feature, {}).get("names")
    if not isinstance(names, list) or len(names) != values.shape[1]:
        names = (
            ["x", "y", "z"]
            if "position" in feature and values.shape[1] == 3
            else [f"{feature}_{i}" for i in range(values.shape[1])]
        )
    unit = "rad" if "joint" in feature else "m" if "ee_position" in feature else "1"
    indices = np.linspace(0, len(values) - 1, min(max_points, len(values)), dtype=int)
    return {
        "feature": feature,
        "names": names,
        "units": [unit] * values.shape[1],
        "timestamps": times[indices].tolist(),
        "values": values[indices].tolist(),
        "point_count": len(indices),
        "total_points": len(values),
        "bounds": {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()},
    }
