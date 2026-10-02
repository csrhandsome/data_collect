"""Convert a measured ROS robot CSV trace into a replay-readable v2.1 dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def import_robot_trace(source: Path, output: Path) -> dict:
    frame = pd.read_csv(source)
    timestamps = frame.timestamp.to_numpy(dtype=float)
    if len(timestamps) < 2 or not np.all(np.isfinite(timestamps)):
        raise ValueError("Trace must contain at least two finite timestamps")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Trace timestamps must increase strictly")
    timestamps -= timestamps[0]
    # Retain real timestamps and select measured samples at approximately 100 Hz.
    indices = np.unique(np.searchsorted(timestamps, np.arange(0, timestamps[-1], 0.01)))
    indices = np.unique(np.append(indices, len(timestamps) - 1))
    frame = frame.iloc[indices]
    timestamp = timestamps[indices]
    joints = frame[[f"q{i}" for i in range(7)]].to_numpy()
    transform = frame[[f"transform{i}" for i in range(16)]].to_numpy()
    commands = frame[[f"command{i}" for i in range(7)]].to_numpy()
    xyz = transform[:, [12, 13, 14]]
    if not all(np.all(np.isfinite(values)) for values in (joints, transform, commands)):
        raise ValueError("Trace contains nonfinite robot state")
    output.mkdir(parents=True, exist_ok=False)
    (output / "meta").mkdir()
    (output / "data/chunk-000").mkdir(parents=True)
    vectors = {"ee_position": xyz, "joint_position": joints, "actions": commands}
    count = len(timestamp)
    table = pa.table(
        {
            "timestamp": timestamp,
            "frame_index": np.arange(count),
            "episode_index": np.zeros(count, dtype=np.int64),
            "index": np.arange(count),
            "task_index": np.zeros(count, dtype=np.int64),
            **{key: values.tolist() for key, values in vectors.items()},
        }
    )
    pq.write_table(table, output / "data/chunk-000/episode_000000.parquet")
    features = {
        name: {
            "dtype": "float64",
            "shape": [values.shape[1]],
            "names": (
                ["x", "y", "z"] if name == "ee_position" else [f"joint_{i}" for i in range(1, 8)]
            ),
        }
        for name, values in vectors.items()
    }
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = {
            "dtype": "float64" if key == "timestamp" else "int64",
            "shape": [1],
            "names": None,
        }
    task = "real Panda small joint-1 round-trip recording"
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "fps": 100,
        "total_episodes": 1,
        "total_frames": count,
        "total_tasks": 1,
        "total_videos": 0,
        "chunks_size": 1000,
        "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "recording_source": str(source.resolve()),
        "synthetic": False,
    }
    (output / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (output / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": count, "tasks": [task]}) + "\n"
    )
    (output / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    summary = {
        "frames": count,
        "duration_s": float(timestamp[-1]),
        "xyz_range_m": np.ptp(xyz, axis=0).tolist(),
        "joint_range_rad": np.ptp(joints, axis=0).tolist(),
        "return_error_rad": (joints[-1] - joints[0]).tolist(),
    }
    (output / "meta/recording_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(import_robot_trace(args.source, args.output), indent=2))
