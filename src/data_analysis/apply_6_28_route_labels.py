#!/usr/bin/env python3
"""Patch route labels and instruction windows for the copied 6_28 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_analysis.instruction_audio_window import (
    DEFAULT_POST_MARGIN_SEC,
    DEFAULT_PRE_MARGIN_SEC,
    build_imputed_instruction_audio_window,
    build_vad_instruction_audio_window,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _episode_index(path: Path) -> int:
    return int(path.name.removeprefix("episode_").split(".")[0])


def _audio_duration(sync_data: dict[str, Any]) -> float | None:
    metadata = sync_data.get("vad_metadata")
    if isinstance(metadata, dict):
        value = metadata.get("audio_duration_sec")
        try:
            return float(value) if value is not None else None
        except Exception:
            return None
    return None


def _sync_paths(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "audio").glob("episode_*.sync.json"))


def _has_vad(sync_data: dict[str, Any]) -> bool:
    segments = sync_data.get("vad_segments")
    return isinstance(segments, list) and len(segments) > 0


def _first_vad_index(sync_by_index: dict[int, dict[str, Any]]) -> int | None:
    for index in sorted(sync_by_index):
        if _has_vad(sync_by_index[index]):
            return index
    return None


def _paired_window(
    *,
    reference_index: int,
    reference_sync: dict[str, Any],
    pre_margin_sec: float,
    post_margin_sec: float,
) -> dict[str, Any] | None:
    window = build_vad_instruction_audio_window(
        reference_sync.get("vad_segments"),
        audio_duration_sec=_audio_duration(reference_sync),
        pre_margin_sec=pre_margin_sec,
        post_margin_sec=post_margin_sec,
    )
    if window is None:
        return None
    return {
        "start_sec": window["start_sec"],
        "end_sec": window["end_sec"],
        "source": "imputed_from_paired_vad",
        "pre_margin_sec": window["pre_margin_sec"],
        "post_margin_sec": window["post_margin_sec"],
        "audio_valid": False,
        "reference_episode_index": int(reference_index),
        "reference_source": "vad",
    }


def apply_labels_and_windows(
    dataset_path: str | Path,
    *,
    straight_count: int = 40,
    pair_source_start: int | None = None,
    pre_margin_sec: float = DEFAULT_PRE_MARGIN_SEC,
    post_margin_sec: float = DEFAULT_POST_MARGIN_SEC,
    dry_run: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_path).resolve()
    if not (dataset_root / "audio").is_dir():
        raise FileNotFoundError(f"Dataset has no audio directory: {dataset_root}")

    paths = _sync_paths(dataset_root)
    sync_by_index = {_episode_index(path): _read_json(path) for path in paths}
    if pair_source_start is None:
        pair_source_start = _first_vad_index(sync_by_index)
    if pair_source_start is None:
        raise RuntimeError("No VAD segments found for paired window references")

    reference_windows: list[dict[str, Any]] = []
    for sync in sync_by_index.values():
        window = build_vad_instruction_audio_window(
            sync.get("vad_segments"),
            audio_duration_sec=_audio_duration(sync),
            pre_margin_sec=pre_margin_sec,
            post_margin_sec=post_margin_sec,
        )
        if window is not None:
            reference_windows.append(window)

    updated = 0
    straight = 0
    detour = 0
    vad_windows = 0
    paired_windows = 0
    median_windows = 0
    pending_windows = 0
    examples: list[dict[str, Any]] = []

    for path in paths:
        index = _episode_index(path)
        sync = sync_by_index[index]
        label = "straight" if index < straight_count else "detour"
        if label == "straight":
            straight += 1
        else:
            detour += 1

        if _has_vad(sync):
            window = build_vad_instruction_audio_window(
                sync.get("vad_segments"),
                audio_duration_sec=_audio_duration(sync),
                pre_margin_sec=pre_margin_sec,
                post_margin_sec=post_margin_sec,
            )
            vad_windows += 1
        elif index < straight_count:
            reference_index = int(pair_source_start) + index
            reference_sync = sync_by_index.get(reference_index)
            window = (
                _paired_window(
                    reference_index=reference_index,
                    reference_sync=reference_sync,
                    pre_margin_sec=pre_margin_sec,
                    post_margin_sec=post_margin_sec,
                )
                if reference_sync is not None
                else None
            )
            if window is not None:
                paired_windows += 1
        else:
            window = None

        if window is None:
            window = build_imputed_instruction_audio_window(
                reference_windows,
                audio_duration_sec=_audio_duration(sync),
                pre_margin_sec=pre_margin_sec,
                post_margin_sec=post_margin_sec,
            )
            if window is not None:
                median_windows += 1
            else:
                pending_windows += 1

        before_label = sync.get("label")
        before_window = sync.get("instruction_audio_window")
        sync["label"] = label
        sync["instruction_audio_window"] = window
        changed = before_label != label or before_window != window
        if changed:
            updated += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "episode_index": index,
                        "label": label,
                        "window_source": window.get("source")
                        if isinstance(window, dict)
                        else None,
                        "start_sec": window.get("start_sec")
                        if isinstance(window, dict)
                        else None,
                        "end_sec": window.get("end_sec")
                        if isinstance(window, dict)
                        else None,
                    }
                )
            if not dry_run:
                _write_json(path, sync)

    return {
        "dataset_path": str(dataset_root),
        "episodes": len(paths),
        "updated": updated,
        "straight": straight,
        "detour": detour,
        "pair_source_start": int(pair_source_start),
        "vad_windows": vad_windows,
        "paired_windows": paired_windows,
        "median_windows": median_windows,
        "pending_windows": pending_windows,
        "reference_window_count": len(reference_windows),
        "dry_run": bool(dry_run),
        "examples": examples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set 6_28 straight/detour labels and instruction windows."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--straight-count", type=int, default=40)
    parser.add_argument(
        "--pair-source-start",
        type=int,
        default=None,
        help="First VAD episode used to provide windows for straight episodes. Defaults to first episode with VAD.",
    )
    parser.add_argument("--pre-margin-sec", type=float, default=DEFAULT_PRE_MARGIN_SEC)
    parser.add_argument("--post-margin-sec", type=float, default=DEFAULT_POST_MARGIN_SEC)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = apply_labels_and_windows(
        args.dataset_path,
        straight_count=int(args.straight_count),
        pair_source_start=args.pair_source_start,
        pre_margin_sec=float(args.pre_margin_sec),
        post_margin_sec=float(args.post_margin_sec),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
