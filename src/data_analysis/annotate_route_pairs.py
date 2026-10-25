#!/usr/bin/env python3
"""Detect route divergence for newly collected straight/detour episode pairs.

The data field policy is intentionally small: collection writes ``label`` and
this script writes ``divergence_time``. Pairing is a script-time concern, not a
stored dataset field. By default, episodes are paired adjacently: 0:1, 2:3, ...;
use ``--pair 3:4`` to specify pairs manually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from data_analysis.instruction_audio_window import instruction_audio_window_end_time


ROUTE_LABELS = {"straight", "detour"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sync_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "audio" / f"episode_{episode_index:06d}.sync.json"


def _episode_indices(dataset_root: Path) -> list[int]:
    info_path = dataset_root / "meta" / "info.json"
    if info_path.is_file():
        total = int(_read_json(info_path).get("total_episodes", 0))
        if total > 0:
            return list(range(total))
    indices: list[int] = []
    for path in sorted((dataset_root / "audio").glob("episode_*.sync.json")):
        try:
            indices.append(int(path.name.removeprefix("episode_").split(".")[0]))
        except ValueError:
            continue
    return indices


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _label(sync: dict[str, Any]) -> str | None:
    value = sync.get("label")
    if value is None:
        value = sync.get("route_label")
    if value is None:
        return None
    text = str(value).strip()
    return text if text in ROUTE_LABELS else None


def _instruction_end_time(sync: dict[str, Any]) -> float:
    end = instruction_audio_window_end_time(sync)
    if end is not None:
        return end
    segments = sync.get("vad_segments")
    if isinstance(segments, list):
        ends = [
            _safe_float(segment.get("end_sec"))
            for segment in segments
            if isinstance(segment, dict)
        ]
        ends = [value for value in ends if value is not None]
        if ends:
            return max(ends)
    return 0.0


def _load_trajectory(
    *,
    sync: dict[str, Any],
    episode_index: int,
    field: str,
    fallback_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    records = sync.get("frame_records")
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError(f"episode {episode_index}: missing frame_records")

    audio_start_ns = sync.get("audio_start_monotonic_ns")
    try:
        audio_start_ns = int(audio_start_ns) if audio_start_ns is not None else None
    except Exception:
        audio_start_ns = None
    fps = _safe_float(sync.get("control_frequency")) or fallback_fps

    times: list[float] = []
    positions: list[list[float]] = []
    for fallback_index, record in enumerate(records):
        if not isinstance(record, dict) or field not in record:
            continue
        position = np.asarray(record[field], dtype=np.float64).reshape(-1)
        if position.size == 0 or not np.all(np.isfinite(position)):
            continue
        frame_index = int(record.get("frame_index", fallback_index))
        host_ns = record.get("host_frame_monotonic_ns")
        try:
            host_ns = int(host_ns) if host_ns is not None else None
        except Exception:
            host_ns = None
        if audio_start_ns is not None and host_ns is not None:
            t = float(host_ns - audio_start_ns) / 1e9
        else:
            t = float(frame_index) / float(fps)
        if np.isfinite(t):
            times.append(t)
            positions.append(position.tolist())

    if len(times) < 2:
        raise ValueError(f"episode {episode_index}: no usable {field} trajectory")
    order = np.argsort(np.asarray(times, dtype=np.float64))
    return np.asarray(times, dtype=np.float64)[order], np.asarray(positions, dtype=np.float64)[order]


def _interp(source_times: np.ndarray, source_positions: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    out = np.empty((target_times.shape[0], source_positions.shape[1]), dtype=np.float64)
    for dim in range(source_positions.shape[1]):
        out[:, dim] = np.interp(target_times, source_times, source_positions[:, dim])
    return out


def _find_divergence_time(
    *,
    a_sync: dict[str, Any],
    b_sync: dict[str, Any],
    a_index: int,
    b_index: int,
    field: str,
    fallback_fps: float,
    threshold: float,
    sustained_sec: float,
) -> tuple[float | None, float]:
    a_times, a_positions = _load_trajectory(
        sync=a_sync, episode_index=a_index, field=field, fallback_fps=fallback_fps
    )
    b_times, b_positions = _load_trajectory(
        sync=b_sync, episode_index=b_index, field=field, fallback_fps=fallback_fps
    )
    if a_positions.shape[1] != b_positions.shape[1]:
        raise ValueError("paired trajectory dimensions do not match")

    start = max(float(a_times[0]), float(b_times[0]))
    end = min(float(a_times[-1]), float(b_times[-1]))
    times = a_times[(a_times >= start) & (a_times <= end)]
    if times.size < 2:
        raise ValueError("paired episodes have too little overlapping time")

    distances = np.linalg.norm(
        _interp(a_times, a_positions, times) - _interp(b_times, b_positions, times),
        axis=1,
    )
    search_start = max(_instruction_end_time(a_sync), _instruction_end_time(b_sync))
    valid_start = int(np.searchsorted(times, search_start, side="left"))
    if valid_start >= times.shape[0]:
        return None, float(np.max(distances))

    dt = float(np.median(np.diff(times))) if times.size >= 2 else 0.0
    index = valid_start
    while index < times.shape[0]:
        if distances[index] <= threshold:
            index += 1
            continue
        end_index = index
        while end_index + 1 < times.shape[0] and distances[end_index + 1] > threshold:
            end_index += 1
        if float(times[end_index] - times[index] + dt) >= sustained_sec:
            return float(times[index]), float(distances[index])
        index = end_index + 1
    return None, float(np.max(distances))


def _parse_pair(value: str) -> tuple[int, int]:
    left, sep, right = value.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError("pair must look like 0:1")
    try:
        return int(left), int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pair indices must be integers") from exc


def _default_pairs(indices: list[int]) -> list[tuple[int, int]]:
    return [(indices[i], indices[i + 1]) for i in range(0, len(indices) - 1, 2)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect paired straight/detour divergence and write divergence_time."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--pair",
        type=_parse_pair,
        action="append",
        default=None,
        help="Explicit episode pair such as 0:1. Repeat for multiple pairs. Defaults to adjacent pairs.",
    )
    parser.add_argument("--trajectory-field", default="ee_position")
    parser.add_argument("--fallback-fps", type=float, default=20.0)
    parser.add_argument("--divergence-threshold", type=float, default=0.03)
    parser.add_argument("--sustained-sec", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = args.dataset_path.resolve()
    if not (dataset_root / "audio").is_dir():
        raise FileNotFoundError(f"Dataset has no audio directory: {dataset_root}")

    indices = _episode_indices(dataset_root)
    pairs = args.pair if args.pair is not None else _default_pairs(indices)
    results: list[dict[str, Any]] = []
    changed: set[int] = set()

    for a_index, b_index in pairs:
        a_path = _sync_path(dataset_root, a_index)
        b_path = _sync_path(dataset_root, b_index)
        if not a_path.is_file() or not b_path.is_file():
            results.append({"pair": f"{a_index}:{b_index}", "status": "missing_sync"})
            continue
        a_sync = _read_json(a_path)
        b_sync = _read_json(b_path)
        labels = {_label(a_sync), _label(b_sync)}
        if labels != ROUTE_LABELS:
            results.append(
                {
                    "pair": f"{a_index}:{b_index}",
                    "status": "bad_labels",
                    "labels": sorted(label for label in labels if label is not None),
                }
            )
            continue
        try:
            divergence_time, distance = _find_divergence_time(
                a_sync=a_sync,
                b_sync=b_sync,
                a_index=a_index,
                b_index=b_index,
                field=str(args.trajectory_field),
                fallback_fps=float(args.fallback_fps),
                threshold=float(args.divergence_threshold),
                sustained_sec=float(args.sustained_sec),
            )
        except Exception as exc:
            results.append({"pair": f"{a_index}:{b_index}", "status": "error", "error": str(exc)})
            continue
        if divergence_time is None:
            results.append(
                {
                    "pair": f"{a_index}:{b_index}",
                    "status": "no_divergence",
                    "max_distance": round(distance, 6),
                }
            )
            continue

        for index, sync in ((a_index, a_sync), (b_index, b_sync)):
            if args.overwrite or sync.get("divergence_time") in (None, ""):
                sync["divergence_time"] = round(divergence_time, 6)
                if not args.dry_run:
                    _write_json(_sync_path(dataset_root, index), sync)
                changed.add(index)
        results.append(
            {
                "pair": f"{a_index}:{b_index}",
                "status": "annotated",
                "divergence_time": round(divergence_time, 6),
                "distance": round(distance, 6),
            }
        )

    print(f"Dataset: {dataset_root}")
    print(f"Pairs checked: {len(pairs)}")
    print(f"Episodes updated: {len(changed)}")
    print(f"Dry run: {bool(args.dry_run)}")
    for result in results:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
