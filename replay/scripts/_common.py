"""Shared, private metadata and path validation for the four reader entry points.

Layouts follow the upstream v0.3.3 (v2.1) and v3 dataset metadata:
https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/datasets/utils.py
https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/dataset_metadata.py
"""

from __future__ import annotations

import json
import math
import string
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_EE_KEYS = {
    "ee_pose",
    "observation.ee_pose",
    "observation.end_effector_pose",
    "ee_position",
    "observation.ee_position",
}
_V2_DATA = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_V2_VIDEO = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_V3_DATA = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_V3_VIDEO = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
_V3_EPISODES = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_PATH_FIELDS = {"episode_chunk", "episode_index", "chunk_index", "file_index", "video_key"}


class MissingDatasetFile(ValueError):
    """A required dataset file is absent; HTTP callers can translate this to 404."""


def _safe_path(root: Path, relative: str | Path, *, require_file: bool = True) -> Path:
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Dataset metadata contains an unsafe path")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Dataset path escapes its root")
    if require_file and not candidate.is_file():
        raise MissingDatasetFile(f"Dataset file not found: {relative}")
    return candidate


def _template_parts(template: Any) -> list[tuple[str, str | None, str, str | None]]:
    if not isinstance(template, str) or not template:
        raise ValueError("Dataset path template must be a nonempty string")
    try:
        parts = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValueError("Malformed dataset path template") from exc
    if any(
        field is not None and (field not in _PATH_FIELDS or conversion)
        for _, field, _, conversion in parts
    ):
        raise ValueError("Unsupported field in dataset path template")
    return parts


def _format_path(root: Path, template: Any, *, require_file: bool = True, **values: Any) -> Path:
    _template_parts(template)
    try:
        relative = template.format(**values)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Cannot format dataset path from episode metadata") from exc
    return _safe_path(root, relative, require_file=require_file)


def _load_json(root: Path, relative: str) -> Any:
    try:
        return json.loads(_safe_path(root, relative).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"Malformed dataset JSON: {relative}") from exc


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _feature_blocks(info: dict) -> list[dict]:
    result = []
    for key, feature in info["features"].items():
        if not isinstance(key, str) or not isinstance(feature, dict):
            raise ValueError("Malformed feature definition")
        dtype, shape, names = feature.get("dtype"), feature.get("shape"), feature.get("names")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            raise ValueError(f"Malformed feature schema: {key}")
        if not shape or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape):
            raise ValueError(f"Invalid feature shape: {key}")
        # Older image datasets used a names dictionary; do not hide their fields.
        if isinstance(names, dict):
            names = next((v for v in names.values() if isinstance(v, list)), None)
        if names is not None and (
            not isinstance(names, list) or not all(isinstance(v, str) for v in names)
        ):
            raise ValueError(f"Invalid feature names: {key}")
        kind = "video" if dtype == "video" else "image" if dtype == "image" else "unsupported"
        if (
            dtype.startswith(("float", "int"))
            and len(shape) == 1
            and key not in {"timestamp", "index", "frame_index", "episode_index", "task_index"}
        ):
            kind = "series"
        if key in _EE_KEYS and len(shape) == 1 and shape[0] in (3, 6, 7):
            kind = "ee"
        labels = {
            "ee_pose": "末端位姿",
            "ee_position": "末端位置",
            "exterior_image_1_left": "外部相机",
            "wrist_image_left": "腕部相机",
        }
        result.append(
            {
                "key": key,
                "label": labels.get(key, key),
                "kind": kind,
                "dtype": dtype,
                "shape": shape,
                "names": names,
            }
        )
    return result


def _read_parquet(path: Path, *, columns: list[str] | None = None) -> pa.Table:
    try:
        return pq.ParquetFile(path).read(columns=columns)
    except (OSError, pa.ArrowException, ValueError) as exc:
        raise ValueError("Malformed dataset parquet file") from exc


def _load_metadata(root: Path) -> tuple[dict, dict[int, dict]]:
    root = Path(root).resolve()
    info = _load_json(root, "meta/info.json")
    if not isinstance(info, dict):
        raise ValueError("Dataset info must be an object")
    version = info.get("codebase_version")
    if version not in {"v2.0", "v2.1", "v3.0"}:
        raise ValueError("Supported dataset versions are v2.0, v2.1 and v3.0")
    _finite_number(info.get("fps"), "fps", positive=True)
    _integer(info.get("total_episodes"), "total_episodes")
    _integer(info.get("total_frames"), "total_frames")
    if not isinstance(info.get("features"), dict):
        raise ValueError("Dataset features must be an object")
    _feature_blocks(info)
    _integer(info.get("chunks_size", 1000), "chunks_size", minimum=1)
    if info.get("storage_format") not in (None, "lerobot"):
        raise ValueError("Only the Parquet/MP4 LeRobot storage format is supported")

    if version.startswith("v2."):
        path = _safe_path(root, "meta/episodes.jsonl")
        try:
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            raise ValueError("Malformed episode JSONL metadata") from exc
    else:
        template = info.get("episodes_path", _V3_EPISODES)
        parts = _template_parts(template)
        if any(
            field is not None and field not in {"chunk_index", "file_index"}
            for _, field, _, _ in parts
        ):
            raise ValueError("Episode metadata template requires chunk/file indices")
        pattern = "".join(
            literal + ("*" if field is not None else "") for literal, field, _, _ in parts
        )
        _safe_path(root, pattern, require_file=False)
        files = sorted(root.glob(pattern))
        if not files and info["total_episodes"]:
            raise MissingDatasetFile("Dataset file not found: episode metadata parquet")
        records = []
        for path in files:
            safe = _safe_path(root, path.relative_to(root))
            file_records = _read_parquet(safe).to_pylist()
            for record in file_records:
                # Upstream records identify the metadata shard as well as data/video shards.
                if any(
                    key in record
                    for key in ("meta/episodes/chunk_index", "meta/episodes/file_index")
                ):
                    chunk = _integer(record.get("meta/episodes/chunk_index"), "metadata chunk")
                    file = _integer(record.get("meta/episodes/file_index"), "metadata file")
                    expected = _format_path(
                        root, template, chunk_index=chunk, file_index=file, require_file=False
                    )
                    if expected != safe:
                        raise ValueError(
                            "Episode metadata chunk/file indices disagree with its path"
                        )
            records.extend(file_records)

    episodes = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Episode metadata must be an object")
        index = _integer(record.get("episode_index"), "episode_index")
        _integer(record.get("length"), "episode length", minimum=1)
        if index in episodes:
            raise ValueError("Duplicate episode index in metadata")
        tasks = record.get("tasks", [])
        if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
            raise ValueError("Episode tasks must be a list of strings")
        if version == "v3.0":
            _integer(record.get("data/chunk_index"), "data chunk")
            _integer(record.get("data/file_index"), "data file")
            first = _integer(record.get("dataset_from_index"), "dataset_from_index")
            last = _integer(record.get("dataset_to_index"), "dataset_to_index")
            if last - first != record["length"]:
                raise ValueError("Episode length disagrees with dataset index boundaries")
        episodes[index] = record
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode count disagrees with info.json")
    if sum(ep["length"] for ep in episodes.values()) != info["total_frames"]:
        raise ValueError("Frame count disagrees with episode metadata")
    return info, dict(sorted(episodes.items()))


def _get_episode(episodes: dict[int, dict], episode_index: int) -> dict:
    _integer(episode_index, "episode_index")
    if episode_index not in episodes:
        raise KeyError(f"Episode {episode_index} not found")
    return episodes[episode_index]


def _episode_summary(info: dict, episode: dict) -> dict:
    return {
        "episode_index": episode["episode_index"],
        "length": episode["length"],
        "duration_s": episode["length"] / info["fps"],
        "tasks": episode.get("tasks", []),
    }


def _episode_table(root: Path, info: dict, episode: dict, feature: str) -> pa.Table:
    index = episode["episode_index"]
    if info["codebase_version"].startswith("v2."):
        path = _format_path(
            root,
            info.get("data_path", _V2_DATA),
            episode_index=index,
            episode_chunk=index // info.get("chunks_size", 1000),
        )
    else:
        path = _format_path(
            root,
            info.get("data_path", _V3_DATA),
            chunk_index=episode["data/chunk_index"],
            file_index=episode["data/file_index"],
        )
    try:
        schema = pq.ParquetFile(path).schema_arrow.names
    except (OSError, pa.ArrowException) as exc:
        raise ValueError("Malformed dataset parquet file") from exc
    if feature not in schema or "episode_index" not in schema:
        raise ValueError("Feature or episode_index column is absent from parquet")
    columns = [
        key
        for key in [feature, "timestamp", "episode_index", "frame_index", "index"]
        if key in schema
    ]
    table = _read_parquet(path, columns=columns)
    try:
        table = table.filter(pc.equal(table["episode_index"], index))
    except pa.ArrowException as exc:
        raise ValueError("Malformed episode_index column") from exc
    if table.num_rows != episode["length"]:
        raise ValueError("Episode parquet row count disagrees with metadata")
    if "frame_index" in columns:
        try:
            table = table.sort_by([("frame_index", "ascending")])
        except pa.ArrowException as exc:
            raise ValueError("Malformed frame_index column") from exc
        if table["frame_index"].to_pylist() != list(range(table.num_rows)):
            raise ValueError("Episode frame_index values must be unique and contiguous")
    if info["codebase_version"] == "v3.0" and "index" in columns:
        if table["index"].to_pylist() != list(
            range(episode["dataset_from_index"], episode["dataset_to_index"])
        ):
            raise ValueError("Episode parquet indices disagree with metadata boundaries")
    return table
