"""Private deterministic signals, illustration rendering, and fixture metadata."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

_FPS, _FRAMES, _EPISODES = 30, 180, 3
_SIZE = (640, 360)
_CAMERAS = ("exterior_image_1_left", "wrist_image_left")
_TASKS = ("Approach the orange block", "Trace a placement arc", "Return above the workbench")
_MARKER = ".replay-demo"
_MARKER_TEXT = "franka-replay synthetic fixture v1\n"


def _features() -> dict:
    scalar = {"shape": [1], "names": None}
    result = {
        "timestamp": {"dtype": "float32", **scalar},
        "frame_index": {"dtype": "int64", **scalar},
        "episode_index": {"dtype": "int64", **scalar},
        "index": {"dtype": "int64", **scalar},
        "task_index": {"dtype": "int64", **scalar},
        "ee_pose": {
            "dtype": "float32",
            "shape": [6],
            "names": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [8],
            "names": [f"joint_{n}" for n in range(1, 8)] + ["gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
        "gripper": {"dtype": "float32", "shape": [1], "names": ["width"]},
        "joint_position": {
            "dtype": "float32",
            "shape": [7],
            "names": [f"joint_{n}" for n in range(1, 8)],
        },
        "gripper_position": {"dtype": "float32", "shape": [1], "names": ["width"]},
        "actions": {
            "dtype": "float32",
            "shape": [7],
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
    }
    for key in _CAMERAS:
        result[key] = {
            "dtype": "video",
            "shape": [_SIZE[1], _SIZE[0], 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": _FPS,
                "video.height": _SIZE[1],
                "video.width": _SIZE[0],
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return result


def _signals(episode: int) -> dict[str, np.ndarray]:
    timestamp = np.arange(_FRAMES, dtype=np.float32) / _FPS
    phase = 2 * np.pi * np.arange(_FRAMES) / (_FRAMES - 1)
    pose = np.column_stack(
        [
            0.45 + 0.09 * np.cos(phase + episode * 0.5),
            -0.10 + episode * 0.07 + 0.10 * np.sin(phase),
            0.32 + 0.045 * np.sin(2 * phase + episode * 0.3),
            np.pi + 0.03 * np.sin(phase),
            0.05 * np.cos(phase),
            0.3 * np.sin(phase + episode * 0.4),
        ]
    ).astype(np.float32)
    gripper = (0.035 + 0.018 * (np.sin(phase - 0.7) + 1) / 2).astype(np.float32)
    joint = np.column_stack([0.4 * np.sin(phase + axis * 0.4) for axis in range(7)])
    return {
        "timestamp": timestamp,
        "frame_index": np.arange(_FRAMES, dtype=np.int64),
        "episode_index": np.full(_FRAMES, episode, dtype=np.int64),
        "index": np.arange(episode * _FRAMES, (episode + 1) * _FRAMES, dtype=np.int64),
        "task_index": np.full(_FRAMES, episode, dtype=np.int64),
        "ee_pose": pose,
        "observation.state": np.column_stack([joint, gripper]).astype(np.float32),
        "action": np.column_stack([pose, gripper]).astype(np.float32),
        "gripper": gripper,
        "joint_position": joint.astype(np.float32),
        "gripper_position": gripper,
        "actions": np.column_stack([pose, gripper]).astype(np.float32),
    }


def _table(signals: dict) -> pa.Table:
    columns = {}
    for key, values in signals.items():
        dtype = pa.int64() if np.issubdtype(values.dtype, np.integer) else pa.float32()
        columns[key] = (
            pa.array(values, type=dtype)
            if values.ndim == 1
            else pa.array(values.tolist(), type=pa.list_(dtype))
        )
    return pa.table(columns)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _write_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _tasks_table() -> pa.Table:
    # Upstream v3 uses a DataFrame indexed by task text with a task_index column.
    # Preserve the pandas index metadata in Arrow without a pandas dependency.
    table = pa.table({"task_index": list(range(_EPISODES)), "__index_level_0__": list(_TASKS)})
    metadata = {
        "index_columns": ["__index_level_0__"],
        "column_indexes": [],
        "columns": [
            {
                "name": "task_index",
                "field_name": "task_index",
                "pandas_type": "int64",
                "numpy_type": "int64",
                "metadata": None,
            },
            {
                "name": None,
                "field_name": "__index_level_0__",
                "pandas_type": "unicode",
                "numpy_type": "object",
                "metadata": None,
            },
        ],
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.2.0",
    }
    return table.replace_schema_metadata({b"pandas": json.dumps(metadata).encode()})


def _stats(signals: dict) -> dict:
    result = {}
    for key, values in signals.items():
        array = values.astype(np.float64)
        if array.ndim == 1:
            array = array[:, None]
        result[key] = {
            "min": array.min(axis=0).tolist(),
            "max": array.max(axis=0).tolist(),
            "mean": array.mean(axis=0).tolist(),
            "std": array.std(axis=0).tolist(),
            "count": [len(array)],
        }
    return result


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _render(episode: int, frame: int, camera: str, pose: np.ndarray) -> np.ndarray:
    image = Image.new("RGB", _SIZE, (12, 20, 29))
    draw = ImageDraw.Draw(image)
    x, y, z = map(float, pose[:3])
    # A perspective workbench with fixed landmarks makes the motion legible.
    draw.polygon([(72, 285), (260, 172), (581, 194), (553, 327)], fill=(34, 48, 58))
    for step in range(9):
        offset = step * 34
        draw.line([(87 + offset, 287), (269 + offset, 173)], fill=(45, 66, 75), width=1)
    for step in range(6):
        draw.line(
            [(96 + step * 27, 270 - step * 14), (559 + step * 4, 317 - step * 23)],
            fill=(45, 66, 75),
            width=1,
        )
    if camera == _CAMERAS[0]:
        tip = (int(352 + (x - 0.45) * 840 + y * 140), int(208 - (z - 0.30) * 760))
        points = [
            (185, 266),
            (185, 223),
            (230, 160),
            (279, 151),
            (int(tip[0] - 32), int(tip[1] - 32)),
            tip,
        ]
        draw.ellipse((148, 252, 222, 281), fill=(10, 16, 22), outline=(89, 110, 121), width=2)
        for start, end in zip(points, points[1:]):
            draw.line([start, end], fill=(98, 122, 133), width=19)
            draw.line(
                [(start[0] - 2, start[1] - 2), (end[0] - 2, end[1] - 2)],
                fill=(204, 215, 216),
                width=11,
            )
        for point in points[1:]:
            draw.ellipse(
                (point[0] - 10, point[1] - 10, point[0] + 10, point[1] + 10),
                fill=(28, 45, 58),
                outline=(150, 174, 184),
                width=3,
            )
        draw.line(
            [(tip[0] - 10, tip[1] + 3), (tip[0] - 10, tip[1] + 22), (tip[0] - 4, tip[1] + 22)],
            fill=(73, 221, 191),
            width=4,
        )
        draw.line(
            [(tip[0] + 10, tip[1] + 3), (tip[0] + 10, tip[1] + 22), (tip[0] + 4, tip[1] + 22)],
            fill=(73, 221, 191),
            width=4,
        )
        bx, by = 400 + episode * 25, 261
    else:
        # Camera translated by EE position: the tabletop target moves under the gripper.
        bx, by = int(320 - (x - 0.45) * 1100), int(215 + (y + 0.04) * 670)
        draw.rectangle((310, 94, 330, 168), fill=(154, 174, 177))
        draw.line([(293, 168), (293, 198)], fill=(73, 221, 191), width=8)
        draw.line([(347, 168), (347, 198)], fill=(73, 221, 191), width=8)
        draw.line([(293, 198), (303, 198)], fill=(73, 221, 191), width=4)
        draw.line([(347, 198), (337, 198)], fill=(73, 221, 191), width=4)
        draw.line([(300, 216), (340, 216)], fill=(64, 119, 124), width=1)
        draw.line([(320, 196), (320, 236)], fill=(64, 119, 124), width=1)
    draw.polygon(
        [(bx, by - 24), (bx + 20, by - 14), (bx, by), (bx - 20, by - 12)], fill=(239, 159, 73)
    )
    draw.polygon(
        [(bx - 20, by - 12), (bx, by), (bx, by + 22), (bx - 20, by + 9)], fill=(166, 92, 43)
    )
    draw.polygon(
        [(bx, by), (bx + 20, by - 14), (bx + 20, by + 8), (bx, by + 22)], fill=(209, 119, 44)
    )
    draw.rectangle((0, 0, 640, 52), fill=(14, 27, 36))
    draw.text(
        (20, 12),
        "SYNTHETIC LAB / " + ("EXTERIOR" if camera == _CAMERAS[0] else "WRIST"),
        font=_font(15),
        fill=(181, 207, 215),
    )
    draw.ellipse((591, 18, 599, 26), fill=(70, 219, 185))
    draw.text(
        (20, 327),
        f"EP {episode:03d}  /  {frame / _FPS:05.2f}s  /  30 FPS",
        font=_font(13),
        fill=(164, 189, 198),
    )
    draw.text((389, 327), f"XYZ  {x:.3f}  {y:.3f}  {z:.3f}", font=_font(12), fill=(74, 210, 183))
    return np.asarray(image)


def _encode_episode(path: Path, episode: int, camera: str, signals: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path),
        _SIZE,
        fps=_FPS,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        quality=8,
        ffmpeg_log_level="error",
        output_params=["-movflags", "+faststart", "-g", str(_FPS), "-threads", "2"],
    )
    writer.send(None)
    minimum, maximum = np.ones(3), np.zeros(3)
    total, square = np.zeros(3), np.zeros(3)
    pixels = 0
    try:
        for frame in range(_FRAMES):
            rgb = _render(episode, frame, camera, signals["ee_pose"][frame])
            writer.send(rgb)
            normalized = rgb.reshape(-1, 3).astype(np.float64) / 255
            minimum = np.minimum(minimum, normalized.min(axis=0))
            maximum = np.maximum(maximum, normalized.max(axis=0))
            total += normalized.sum(axis=0)
            square += np.square(normalized).sum(axis=0)
            pixels += len(normalized)
    finally:
        writer.close()
    mean = total / pixels
    std = np.sqrt(np.maximum(square / pixels - mean**2, 0))
    # LeRobot video statistics use channel-first singleton spatial dimensions.
    return {
        "min": minimum[:, None, None].tolist(),
        "max": maximum[:, None, None].tolist(),
        "mean": mean[:, None, None].tolist(),
        "std": std[:, None, None].tolist(),
        "count": [_FRAMES],
    }


def _concat_videos(sources: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="replay-concat-") as directory:
        listing = Path(directory) / "inputs.txt"
        listing.write_text(
            "".join("file '" + str(path).replace("'", "'\\''") + "'\n" for path in sources),
            encoding="utf-8",
        )
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(destination),
            ],
            check=True,
            capture_output=True,
        )


def _aggregate_video_stats(episodes: list[dict], key: str) -> dict:
    values = [episode[key] for episode in episodes]
    mean = np.mean([value["mean"] for value in values], axis=0)
    variance = (
        np.mean(
            [np.asarray(value["std"]) ** 2 + np.asarray(value["mean"]) ** 2 for value in values],
            axis=0,
        )
        - mean**2
    )
    return {
        "min": np.min([value["min"] for value in values], axis=0).tolist(),
        "max": np.max([value["max"] for value in values], axis=0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(variance, 0)).tolist(),
        "count": [_FRAMES * _EPISODES],
    }


def _info(version: str) -> dict:
    return {
        "name": "Franka workbench · " + version,
        "codebase_version": version,
        "robot_type": "franka",
        "fps": _FPS,
        "total_episodes": _EPISODES,
        "total_frames": _FRAMES * _EPISODES,
        "total_tasks": _EPISODES,
        "total_videos": len(_CAMERAS) * (_EPISODES if version == "v2.1" else 1),
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:3"},
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            if version == "v2.1"
            else "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            if version == "v2.1"
            else "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
        "features": _features(),
        "synthetic": True,
    }
