"""Target layout for the project's LeRobot v3 MP4 datasets."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v3.0"
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

# The two gripper slots are this project's internal finger RGB cameras.
# Source visuo-tactile video is written only into these two keys.
GRIPPER_IMAGE_LEFT = "observation.gripper_image_left"
GRIPPER_IMAGE_RIGHT = "observation.gripper_image_right"

# Names match control.collection.schema. Video keys are dtype video; the rest are vectors.
FEATURES = {
    "observation.exterior_image": {"dtype": "video", "shape": None, "names": ["height", "width", "channel"]},
    "observation.wrist_image_left": {"dtype": "video", "shape": None, "names": ["height", "width", "channel"]},
    GRIPPER_IMAGE_LEFT: {"dtype": "video", "shape": None, "names": ["height", "width", "channel"]},
    GRIPPER_IMAGE_RIGHT: {"dtype": "video", "shape": None, "names": ["height", "width", "channel"]},
    "observation.joint_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": [f"joint_{i}" for i in range(7)],
    },
    "observation.ee_pose": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw"],
    },
    "observation.ee_position": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
    "observation.gripper_position": {"dtype": "float32", "shape": (1,), "names": ["commanded_open_ratio"]},
    "action.joint_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": [f"joint_{i}" for i in range(7)],
    },
    "action.ee_pose": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw"],
    },
    "action.gripper_position": {"dtype": "float32", "shape": (1,), "names": ["commanded_open_ratio"]},
}


@dataclass
class AlignedEpisode:
    """One source episode already mapped onto the v3 feature names."""

    key: str
    task: str
    fps: float
    videos: dict[str, Path] = field(default_factory=dict)
    columns: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = [name for name in (*self.videos, *self.columns) if name not in FEATURES]
        if unknown:
            raise ValueError(f"unknown v3 features: {unknown}")
        lengths = [len(value) for value in self.columns.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"{self.key} columns disagree on length: {lengths}")

    @property
    def length(self) -> int:
        if self.columns:
            return len(next(iter(self.columns.values())))
        return 0


def _video_shape(path: Path) -> tuple[int, int, int]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    text = (probe.stdout or "").strip()
    if probe.returncode == 0 and "x" in text:
        width, height = text.split("x", 1)
        if width.isdigit() and height.isdigit():
            return int(height), int(width), 3
    return 480, 640, 3


def _vector(values: np.ndarray, width: int) -> pa.Array:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[1] != width:
        raise ValueError(f"expected width {width}, got {array.shape}")
    flat = pa.array(array.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def write_dataset(episodes: list[AlignedEpisode], output: Path, *, robot_type: str, fps: float) -> Path:
    """Write a small v3 dataset. Callers pass one trial episode when disk is tight."""
    if not episodes:
        raise ValueError("no episodes to write")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    fps_value = int(round(float(fps)))
    if fps_value < 1:
        raise ValueError(f"fps must be positive, got {fps}")

    video_keys = list(dict.fromkeys(key for episode in episodes for key in episode.videos))
    column_keys = list(dict.fromkeys(key for episode in episodes for key in episode.columns))
    for episode in episodes:
        if set(episode.videos) != set(video_keys) or set(episode.columns) != set(column_keys):
            raise ValueError(f"{episode.key} does not match the other episodes' features")

    data_dir = output / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    video_shapes: dict[str, tuple[int, int, int]] = {}
    for key in video_keys:
        destination = output / "videos" / key / "chunk-000" / "file-000.mp4"
        destination.parent.mkdir(parents=True)
        shutil.copyfile(episodes[0].videos[key], destination)
        for episode in episodes[1:]:
            # One shared MP4 per feature keeps the trial small. Extra episodes would need a concat.
            if episode.videos[key].read_bytes() != destination.read_bytes():
                raise ValueError("trial writer currently stores one video file; pass a single episode")
        video_shapes[key] = _video_shape(destination)

    rows: dict[str, list] = {
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
    }
    numeric = {key: [] for key in column_keys}
    episode_rows = []
    tasks: list[str] = []
    cursor = 0
    for episode_index, episode in enumerate(episodes):
        task = episode.task or episode.key
        if task not in tasks:
            tasks.append(task)
        task_index = tasks.index(task)
        length = episode.length
        if length < 1:
            raise ValueError(f"{episode.key} has no frames")
        rows["timestamp"].extend((np.arange(length, dtype=np.float32) / fps_value).tolist())
        rows["frame_index"].extend(range(length))
        rows["episode_index"].extend([episode_index] * length)
        rows["index"].extend(range(cursor, cursor + length))
        rows["task_index"].extend([task_index] * length)
        for key in column_keys:
            numeric[key].append(episode.columns[key])
        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [task],
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": cursor,
                "dataset_to_index": cursor + length,
            }
        )
        cursor += length

    table = pa.table(
        {
            "timestamp": pa.array(rows["timestamp"], type=pa.float32()),
            "frame_index": pa.array(rows["frame_index"], type=pa.int64()),
            "episode_index": pa.array(rows["episode_index"], type=pa.int64()),
            "index": pa.array(rows["index"], type=pa.int64()),
            "task_index": pa.array(rows["task_index"], type=pa.int64()),
            **{
                key: _vector(np.concatenate(numeric[key], axis=0), FEATURES[key]["shape"][0])
                for key in column_keys
            },
        }
    )
    pq.write_table(table, data_dir / "file-000.parquet")

    episodes_dir = output / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episode_rows), episodes_dir / "file-000.parquet")
    pq.write_table(
        pa.table({"task_index": pa.array(range(len(tasks)), type=pa.int64())}),
        output / "meta" / "tasks.parquet",
    )

    features = {}
    for key in video_keys:
        height, width, channels = video_shapes[key]
        features[key] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": fps_value,
                "video.height": height,
                "video.width": width,
                "video.channels": channels,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    for key in column_keys:
        spec = FEATURES[key]
        features[key] = {"dtype": spec["dtype"], "shape": list(spec["shape"]), "names": list(spec["names"])}
    for key, spec in {
        "timestamp": ("float32", [1]),
        "frame_index": ("int64", [1]),
        "episode_index": ("int64", [1]),
        "index": ("int64", [1]),
        "task_index": ("int64", [1]),
    }.items():
        features[key] = {"dtype": spec[0], "shape": spec[1], "names": None}

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "fps": fps_value,
        "features": features,
        "total_episodes": len(episodes),
        "total_frames": cursor,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "splits": {"train": f"0:{len(episodes)}"},
    }
    (output / "meta").mkdir(exist_ok=True)
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    (output / "meta" / "stats.json").write_text("{}\n")
    return output
