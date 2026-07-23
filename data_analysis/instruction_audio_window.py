#!/usr/bin/env python3
"""Derive instruction audio windows from VAD metadata.

The raw VAD result stays in ``vad_segments``.  This module writes the derived
``instruction_audio_window`` used for audio cropping and action alignment.
Episodes without VAD can be aligned with the dataset median from episodes that
do have VAD, while still being marked as imputed rather than real speech.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_PRE_MARGIN_SEC = 0.2
DEFAULT_POST_MARGIN_SEC = 0.4


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except Exception:
        return None
    return out if out == out and abs(out) != float("inf") else None


def _round_time(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _clip_time(value: float, *, lower: float = 0.0, upper: float | None = None) -> float:
    clipped = max(float(lower), float(value))
    if upper is not None:
        clipped = min(clipped, float(upper))
    return clipped


def make_pending_instruction_audio_window(
    *,
    pre_margin_sec: float = DEFAULT_PRE_MARGIN_SEC,
    post_margin_sec: float = DEFAULT_POST_MARGIN_SEC,
) -> dict[str, Any]:
    return {
        "start_sec": None,
        "end_sec": None,
        "source": "pending_vad",
        "pre_margin_sec": round(float(pre_margin_sec), 6),
        "post_margin_sec": round(float(post_margin_sec), 6),
        "audio_valid": False,
    }


def build_vad_instruction_audio_window(
    vad_segments: Any,
    *,
    audio_duration_sec: float | None = None,
    pre_margin_sec: float = DEFAULT_PRE_MARGIN_SEC,
    post_margin_sec: float = DEFAULT_POST_MARGIN_SEC,
) -> dict[str, Any] | None:
    if not isinstance(vad_segments, list):
        return None

    starts: list[float] = []
    ends: list[float] = []
    for segment in vad_segments:
        if not isinstance(segment, dict):
            continue
        start = _safe_float(segment.get("start_sec"))
        end = _safe_float(segment.get("end_sec"))
        if start is None or end is None or end <= start:
            continue
        starts.append(start)
        ends.append(end)

    if not starts or not ends:
        return None

    raw_start = min(starts)
    raw_end = max(ends)
    duration = _safe_float(audio_duration_sec)
    start_sec = _clip_time(raw_start - float(pre_margin_sec), upper=duration)
    end_sec = _clip_time(raw_end + float(post_margin_sec), upper=duration)
    if end_sec < start_sec:
        end_sec = start_sec

    return {
        "start_sec": _round_time(start_sec),
        "end_sec": _round_time(end_sec),
        "source": "vad",
        "pre_margin_sec": round(float(pre_margin_sec), 6),
        "post_margin_sec": round(float(post_margin_sec), 6),
        "audio_valid": True,
        "vad_start_sec": _round_time(raw_start),
        "vad_end_sec": _round_time(raw_end),
        "num_vad_segments": len(starts),
    }


def build_imputed_instruction_audio_window(
    reference_windows: list[dict[str, Any]],
    *,
    audio_duration_sec: float | None = None,
    pre_margin_sec: float = DEFAULT_PRE_MARGIN_SEC,
    post_margin_sec: float = DEFAULT_POST_MARGIN_SEC,
) -> dict[str, Any] | None:
    starts: list[float] = []
    ends: list[float] = []
    for window in reference_windows:
        if not isinstance(window, dict):
            continue
        start = _safe_float(window.get("start_sec"))
        end = _safe_float(window.get("end_sec"))
        if start is None or end is None or end < start:
            continue
        starts.append(start)
        ends.append(end)

    if not starts or not ends:
        return None

    duration = _safe_float(audio_duration_sec)
    start_sec = _clip_time(float(median(starts)), upper=duration)
    end_sec = _clip_time(float(median(ends)), upper=duration)
    if end_sec < start_sec:
        end_sec = start_sec

    return {
        "start_sec": _round_time(start_sec),
        "end_sec": _round_time(end_sec),
        "source": "imputed_dataset_median",
        "pre_margin_sec": round(float(pre_margin_sec), 6),
        "post_margin_sec": round(float(post_margin_sec), 6),
        "audio_valid": False,
        "reference_window_count": len(starts),
    }


def instruction_audio_window_end_time(sync_data: dict[str, Any]) -> float | None:
    for key in (
        "instruction_audio_window",
        "instruction_window",
        "pseudo_instruction_window",
    ):
        window = sync_data.get(key)
        if isinstance(window, dict):
            end = _safe_float(window.get("end_sec"))
            if end is not None:
                return end
    return _safe_float(sync_data.get("instruction_end_time"))


def _episode_indices(dataset_root: Path) -> list[int]:
    info_path = dataset_root / "meta" / "info.json"
    if info_path.is_file():
        try:
            total = int(_read_json(info_path).get("total_episodes", 0))
        except Exception:
            total = 0
        if total > 0:
            return list(range(total))

    indices: list[int] = []
    for path in sorted((dataset_root / "audio").glob("episode_*.sync.json")):
        try:
            indices.append(int(path.name.removeprefix("episode_").split(".")[0]))
        except ValueError:
            continue
    return indices


def _sync_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "audio" / f"episode_{episode_index:06d}.sync.json"


def _audio_duration(sync_data: dict[str, Any]) -> float | None:
    metadata = sync_data.get("vad_metadata")
    if isinstance(metadata, dict):
        duration = _safe_float(metadata.get("audio_duration_sec"))
        if duration is not None:
            return duration
    return _safe_float(sync_data.get("audio_duration_sec"))


def refresh_dataset_instruction_audio_windows(
    dataset_path: str | Path,
    *,
    episode_indices: list[int] | None = None,
    pre_margin_sec: float = DEFAULT_PRE_MARGIN_SEC,
    post_margin_sec: float = DEFAULT_POST_MARGIN_SEC,
    overwrite: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_path).resolve()
    all_indices = _episode_indices(dataset_root)
    target_indices = episode_indices or all_indices

    sync_by_index: dict[int, dict[str, Any]] = {}
    reference_windows: list[dict[str, Any]] = []
    for index in all_indices:
        sync = _read_json(_sync_path(dataset_root, int(index)))
        sync_by_index[int(index)] = sync
        window = build_vad_instruction_audio_window(
            sync.get("vad_segments"),
            audio_duration_sec=_audio_duration(sync),
            pre_margin_sec=pre_margin_sec,
            post_margin_sec=post_margin_sec,
        )
        if window is not None:
            reference_windows.append(window)

    updated = 0
    vad_windows = 0
    imputed_windows = 0
    pending_windows = 0
    missing_sync = 0
    for index in target_indices:
        index = int(index)
        path = _sync_path(dataset_root, index)
        sync = sync_by_index.get(index) or _read_json(path)
        if not path.is_file() and not sync:
            missing_sync += 1
            continue
        if not overwrite and isinstance(sync.get("instruction_audio_window"), dict):
            continue

        duration = _audio_duration(sync)
        window = build_vad_instruction_audio_window(
            sync.get("vad_segments"),
            audio_duration_sec=duration,
            pre_margin_sec=pre_margin_sec,
            post_margin_sec=post_margin_sec,
        )
        if window is not None:
            vad_windows += 1
        else:
            window = build_imputed_instruction_audio_window(
                reference_windows,
                audio_duration_sec=duration,
                pre_margin_sec=pre_margin_sec,
                post_margin_sec=post_margin_sec,
            )
            if window is not None:
                imputed_windows += 1
            else:
                window = make_pending_instruction_audio_window(
                    pre_margin_sec=pre_margin_sec,
                    post_margin_sec=post_margin_sec,
                )
                pending_windows += 1

        if sync.get("instruction_audio_window") == window:
            continue
        sync["instruction_audio_window"] = window
        updated += 1
        if not dry_run:
            _write_json(path, sync)

    return {
        "dataset_path": str(dataset_root),
        "episodes": len(target_indices),
        "updated": updated,
        "missing_sync": missing_sync,
        "vad_windows": vad_windows,
        "imputed_windows": imputed_windows,
        "pending_windows": pending_windows,
        "reference_window_count": len(reference_windows),
        "dry_run": bool(dry_run),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh instruction_audio_window from VAD metadata."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, action="append", default=None)
    parser.add_argument("--pre-margin-sec", type=float, default=DEFAULT_PRE_MARGIN_SEC)
    parser.add_argument("--post-margin-sec", type=float, default=DEFAULT_POST_MARGIN_SEC)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = refresh_dataset_instruction_audio_windows(
        args.dataset_path,
        episode_indices=args.episode_index,
        pre_margin_sec=float(args.pre_margin_sec),
        post_margin_sec=float(args.post_margin_sec),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
