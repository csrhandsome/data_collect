#!/usr/bin/env python3
"""Offline Silero-VAD preprocessing for per-episode audio.

This script runs after data collection and before training. It reads the existing
``audio/episode_XXXXXX.wav`` files, detects speech segments, optionally writes
short WAV clips, and stores the result back into the existing
``audio/episode_XXXXXX.sync.json`` sidecar under ``vad_segments``.

Example:
  uv run -m data_analysis.preprocess_vad \
    --dataset-path data/openpi/franka_lerobot_4_9_audio \
    --overwrite
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable
import wave

import numpy as np

from data_analysis.instruction_audio_window import (
    DEFAULT_POST_MARGIN_SEC,
    DEFAULT_PRE_MARGIN_SEC,
    refresh_dataset_instruction_audio_windows,
)
from control.util.audio_util import read_wav_pcm


SpeechTimestampFn = Callable[..., list[dict[str, int]]]


def _path_for_json(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _json_float_or_none(value: float) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _resolve_audio_path(
    dataset_root: Path, sync_data: dict[str, Any], default_path: Path
) -> Path:
    raw_path = sync_data.get("audio_path")
    if not raw_path:
        return default_path
    audio_path = Path(str(raw_path))
    if audio_path.is_absolute():
        return audio_path
    return dataset_root / audio_path


def _episode_indices(dataset_root: Path) -> list[int]:
    info_path = dataset_root / "meta" / "info.json"
    if info_path.is_file():
        info = _read_json(info_path)
        total = int(info.get("total_episodes", 0))
        if total > 0:
            return list(range(total))

    audio_dir = dataset_root / "audio"
    indices: list[int] = []
    for wav_path in sorted(audio_dir.glob("episode_*.wav")):
        try:
            indices.append(int(wav_path.stem.split("_")[-1]))
        except ValueError:
            continue
    return indices


def _audio_to_mono_float(audio: np.ndarray) -> np.ndarray:
    pcm = np.asarray(audio)
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=1)
    elif pcm.ndim > 2:
        raise ValueError(f"Expected mono/stereo audio, got shape {pcm.shape}")

    if np.issubdtype(pcm.dtype, np.integer):
        dtype_info = np.iinfo(pcm.dtype)
        if pcm.dtype.kind == "u":
            midpoint = float(dtype_info.max + 1) / 2.0
            return ((pcm.astype(np.float32) - midpoint) / midpoint).astype(np.float32)
        scale = max(float(dtype_info.max), float(-dtype_info.min), 1.0)
        return (pcm.astype(np.float32) / scale).astype(np.float32)

    return np.clip(pcm.astype(np.float32), -1.0, 1.0)


def _resample_linear(
    audio: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate == target_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.ascontiguousarray(audio, dtype=np.float32)

    duration_s = float(audio.shape[0]) / float(source_rate)
    target_len = max(1, int(round(duration_s * float(target_rate))))
    source_x = np.linspace(0.0, duration_s, num=audio.shape[0], endpoint=False)
    target_x = np.linspace(0.0, duration_s, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, audio).astype(np.float32)


def _prepare_vad_audio(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    mono = _audio_to_mono_float(audio)
    if sample_rate in (8000, 16000):
        return np.ascontiguousarray(mono, dtype=np.float32), int(sample_rate)

    # Silero supports 8 kHz and 16 kHz. The collector defaults to 16 kHz, but
    # this fallback keeps older/off-nominal recordings usable.
    return _resample_linear(mono, int(sample_rate), 16000), 16000


def _write_wav_pcm(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clip = np.asarray(audio)
    if clip.ndim == 1:
        clip = clip.reshape(-1, 1)
    if clip.ndim != 2:
        raise ValueError(f"Expected clip shape (samples, channels), got {clip.shape}")

    if np.issubdtype(clip.dtype, np.floating):
        clip = np.clip(clip, -1.0, 1.0)
        clip = np.rint(clip * np.iinfo(np.int16).max).astype(np.int16)
    elif clip.dtype not in (np.uint8, np.int16, np.int32):
        clip = clip.astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(int(clip.shape[1]))
        wav_file.setsampwidth(np.dtype(clip.dtype).itemsize)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(np.ascontiguousarray(clip).tobytes())


def _find_silero_jit_model_path() -> Path:
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("silero-vad is not installed in the current environment")
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    model_path = package_dir / "data" / "silero_vad.jit"
    if not model_path.is_file():
        raise FileNotFoundError(f"Silero VAD JIT model not found: {model_path}")
    return model_path


def _fallback_get_speech_timestamps(
    audio,
    model,
    *,
    threshold: float,
    sampling_rate: int,
    min_speech_duration_ms: int,
    max_speech_duration_s: float,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    neg_threshold: float | None = None,
    **_: Any,
) -> list[dict[str, int]]:
    import torch

    if not torch.is_tensor(audio):
        wav = torch.as_tensor(audio, dtype=torch.float32)
    else:
        wav = audio.to(dtype=torch.float32)
    wav = wav.flatten()

    if int(sampling_rate) not in (8000, 16000):
        raise ValueError("Silero VAD fallback supports only 8000 or 16000 Hz audio")

    window_size = 512 if int(sampling_rate) == 16000 else 256
    model.reset_states()
    speech_probs: list[float] = []
    with torch.no_grad():
        for start in range(0, int(wav.numel()), window_size):
            chunk = wav[start : start + window_size]
            if int(chunk.numel()) < window_size:
                chunk = torch.nn.functional.pad(
                    chunk, (0, window_size - int(chunk.numel()))
                )
            speech_probs.append(float(model(chunk, int(sampling_rate)).item()))

    neg = (
        max(float(threshold) - 0.15, 0.01)
        if neg_threshold is None
        else float(neg_threshold)
    )
    min_speech_samples = int(
        round(int(sampling_rate) * min_speech_duration_ms / 1000.0)
    )
    min_silence_samples = int(
        round(int(sampling_rate) * min_silence_duration_ms / 1000.0)
    )
    pad_samples = int(round(int(sampling_rate) * speech_pad_ms / 1000.0))
    max_speech_samples = (
        None
        if not np.isfinite(max_speech_duration_s)
        else int(round(int(sampling_rate) * float(max_speech_duration_s)))
    )

    segments: list[dict[str, int]] = []
    triggered = False
    speech_start = 0
    silence_start: int | None = None

    for index, prob in enumerate(speech_probs):
        cur_sample = index * window_size
        if prob >= float(threshold):
            if not triggered:
                triggered = True
                speech_start = cur_sample
            silence_start = None
        elif triggered and prob < neg:
            if silence_start is None:
                silence_start = cur_sample
            if cur_sample - silence_start >= min_silence_samples:
                speech_end = silence_start
                if speech_end - speech_start >= min_speech_samples:
                    segments.append({"start": speech_start, "end": speech_end})
                triggered = False
                silence_start = None

        if (
            triggered
            and max_speech_samples is not None
            and cur_sample - speech_start >= max_speech_samples
        ):
            segments.append({"start": speech_start, "end": cur_sample})
            triggered = False
            silence_start = None

    if triggered:
        speech_end = int(wav.numel())
        if speech_end - speech_start >= min_speech_samples:
            segments.append({"start": speech_start, "end": speech_end})

    if not segments:
        return []

    padded: list[dict[str, int]] = []
    audio_len = int(wav.numel())
    for segment in segments:
        start = max(0, int(segment["start"]) - pad_samples)
        end = min(audio_len, int(segment["end"]) + pad_samples)
        if padded and start <= padded[-1]["end"]:
            padded[-1]["end"] = max(padded[-1]["end"], end)
        else:
            padded.append({"start": start, "end": end})
    return padded


def _load_silero() -> tuple[Any, SpeechTimestampFn, dict[str, Any]]:
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad()
        loader = "silero_vad.load_silero_vad"
        timestamp_fn = get_speech_timestamps
    except Exception:
        import torch

        model_path = _find_silero_jit_model_path()
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        loader = "torch.jit.load"
        timestamp_fn = _fallback_get_speech_timestamps

    try:
        version = importlib.metadata.version("silero-vad")
    except importlib.metadata.PackageNotFoundError:
        version = None

    return (
        model,
        timestamp_fn,
        {
            "tool": "silero-vad",
            "tool_version": version,
            "model_loader": loader,
        },
    )


def _detect_segments(
    *,
    audio: np.ndarray,
    sample_rate: int,
    model: Any,
    timestamp_fn: SpeechTimestampFn,
    threshold: float,
    min_speech_duration_ms: int,
    max_speech_duration_s: float,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    neg_threshold: float | None,
) -> tuple[list[dict[str, int]], int]:
    import torch

    vad_audio, vad_sample_rate = _prepare_vad_audio(audio, sample_rate)
    timestamps = timestamp_fn(
        torch.from_numpy(vad_audio),
        model,
        threshold=float(threshold),
        sampling_rate=int(vad_sample_rate),
        min_speech_duration_ms=int(min_speech_duration_ms),
        max_speech_duration_s=float(max_speech_duration_s),
        min_silence_duration_ms=int(min_silence_duration_ms),
        speech_pad_ms=int(speech_pad_ms),
        return_seconds=False,
        neg_threshold=neg_threshold,
    )
    normalized = [
        {"start": int(item["start"]), "end": int(item["end"])}
        for item in timestamps
        if int(item.get("end", 0)) > int(item.get("start", 0))
    ]
    return normalized, int(vad_sample_rate)


def _segment_payloads(
    *,
    dataset_root: Path,
    episode_index: int,
    audio: np.ndarray,
    sample_rate: int,
    vad_sample_rate: int,
    timestamps: list[dict[str, int]],
    clips_dir: Path | None,
) -> list[dict[str, Any]]:
    total_samples = int(audio.shape[0])
    segments: list[dict[str, Any]] = []
    for segment_index, timestamp in enumerate(timestamps):
        start_sec = float(timestamp["start"]) / float(vad_sample_rate)
        end_sec = float(timestamp["end"]) / float(vad_sample_rate)
        start_sample = int(np.clip(round(start_sec * sample_rate), 0, total_samples))
        end_sample = int(
            np.clip(round(end_sec * sample_rate), start_sample, total_samples)
        )
        if end_sample <= start_sample:
            continue

        clip_path: Path | None = None
        if clips_dir is not None:
            clip_path = clips_dir / f"seg_{segment_index:03d}.wav"
            _write_wav_pcm(clip_path, audio[start_sample:end_sample], sample_rate)

        segments.append(
            {
                "seg_id": f"episode_{episode_index:06d}_vad_{segment_index:03d}",
                "source": "silero-vad",
                "kind": "instruction",
                "start_sec": round(start_sec, 6),
                "end_sec": round(end_sec, 6),
                "duration_sec": round(max(0.0, end_sec - start_sec), 6),
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "audio_wav_path": _path_for_json(clip_path, dataset_root),
                "asr_transcript": None,
                "verified": False,
            }
        )
    return segments


def process_episode(
    *,
    dataset_root: Path,
    episode_index: int,
    model: Any,
    timestamp_fn: SpeechTimestampFn,
    model_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    audio_dir = dataset_root / "audio"
    stem = f"episode_{episode_index:06d}"
    default_wav_path = audio_dir / f"{stem}.wav"
    sync_path = audio_dir / f"{stem}.sync.json"
    sync_data = _read_json(sync_path)

    if not bool(args.overwrite) and bool(
        sync_data.get("vad_metadata", {}).get("processed")
    ):
        return {
            "episode_index": episode_index,
            "status": "skipped",
            "reason": "already_processed",
            "sync_path": str(sync_path),
        }

    wav_path = _resolve_audio_path(dataset_root, sync_data, default_wav_path)
    if not wav_path.is_file():
        return {
            "episode_index": episode_index,
            "status": "missing_audio",
            "audio_path": str(wav_path),
        }

    audio, sample_rate = read_wav_pcm(wav_path, require_nonempty=True)
    timestamps, vad_sample_rate = _detect_segments(
        audio=audio,
        sample_rate=int(sample_rate),
        model=model,
        timestamp_fn=timestamp_fn,
        threshold=float(args.threshold),
        min_speech_duration_ms=int(args.min_speech_duration_ms),
        max_speech_duration_s=float(args.max_speech_duration_s),
        min_silence_duration_ms=int(args.min_silence_duration_ms),
        speech_pad_ms=int(args.speech_pad_ms),
        neg_threshold=args.neg_threshold,
    )

    clips_dir = None
    if bool(args.save_clips):
        clips_dir = Path(args.clips_dir)
        if not clips_dir.is_absolute():
            clips_dir = dataset_root / clips_dir
        clips_dir = clips_dir / stem

    segments = _segment_payloads(
        dataset_root=dataset_root,
        episode_index=episode_index,
        audio=audio,
        sample_rate=int(sample_rate),
        vad_sample_rate=int(vad_sample_rate),
        timestamps=timestamps,
        clips_dir=clips_dir,
    )

    sync_data.update(
        {
            "episode_index": int(sync_data.get("episode_index", episode_index)),
            "audio_path": _path_for_json(wav_path, dataset_root),
            "vad_metadata": {
                "processed": True,
                "stage": "offline_preprocess",
                **model_info,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "audio_sample_rate": int(sample_rate),
                "audio_num_samples": int(audio.shape[0]),
                "audio_duration_sec": round(float(audio.shape[0]) / float(sample_rate), 6),
                "vad_sample_rate": int(vad_sample_rate),
                "num_segments": len(segments),
                "segments_dir": _path_for_json(clips_dir, dataset_root),
                "parameters": {
                    "threshold": float(args.threshold),
                    "neg_threshold": args.neg_threshold,
                    "min_speech_duration_ms": int(args.min_speech_duration_ms),
                    "max_speech_duration_s": _json_float_or_none(
                        args.max_speech_duration_s
                    ),
                    "min_silence_duration_ms": int(args.min_silence_duration_ms),
                    "speech_pad_ms": int(args.speech_pad_ms),
                    "save_clips": bool(args.save_clips),
                },
            },
            "vad_segments": segments,
        }
    )
    _write_json(sync_path, sync_data)

    return {
        "episode_index": episode_index,
        "status": "processed",
        "num_segments": len(segments),
        "sync_path": str(sync_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline Silero-VAD and write vad_segments into episode sync JSON."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/openpi/franka_lerobot_4_9_audio"),
        help="LeRobot dataset root containing audio/ and meta/.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        action="append",
        default=None,
        help="Episode index to process. Repeat to process multiple episodes. Defaults to all episodes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run VAD even if vad_metadata.processed is already true.",
    )
    parser.add_argument(
        "--save-clips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write each detected VAD segment as a WAV clip under --clips-dir.",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=Path("audio/vad_segments"),
        help="Directory for VAD clips, relative to dataset root unless absolute.",
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--neg-threshold", type=float, default=None)
    parser.add_argument("--min-speech-duration-ms", type=int, default=250)
    parser.add_argument("--max-speech-duration-s", type=float, default=float("inf"))
    parser.add_argument("--min-silence-duration-ms", type=int, default=100)
    parser.add_argument("--speech-pad-ms", type=int, default=30)
    parser.add_argument(
        "--no-instruction-audio-window",
        dest="instruction_audio_window",
        action="store_false",
        default=True,
        help="Skip deriving instruction_audio_window after VAD.",
    )
    parser.add_argument(
        "--instruction-pre-margin-sec",
        type=float,
        default=DEFAULT_PRE_MARGIN_SEC,
    )
    parser.add_argument(
        "--instruction-post-margin-sec",
        type=float,
        default=DEFAULT_POST_MARGIN_SEC,
    )
    return parser


def compute_vad_for_dataset(
    dataset_path: str | Path,
    *,
    episode_indices: list[int] | None = None,
    overwrite: bool = False,
    save_clips: bool = True,
    clips_dir: str | Path = Path("audio/vad_segments"),
    threshold: float = 0.5,
    neg_threshold: float | None = None,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    instruction_audio_window: bool = True,
    instruction_pre_margin_sec: float = DEFAULT_PRE_MARGIN_SEC,
    instruction_post_margin_sec: float = DEFAULT_POST_MARGIN_SEC,
    print_summary: bool = True,
) -> dict[str, Any]:
    """Run offline VAD for a LeRobot dataset and write episode sync JSON files."""

    args = argparse.Namespace(
        overwrite=overwrite,
        save_clips=save_clips,
        clips_dir=Path(clips_dir),
        threshold=threshold,
        neg_threshold=neg_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        instruction_audio_window=instruction_audio_window,
        instruction_pre_margin_sec=instruction_pre_margin_sec,
        instruction_post_margin_sec=instruction_post_margin_sec,
    )

    dataset_root = Path(dataset_path).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")
    if not (dataset_root / "audio").is_dir():
        raise FileNotFoundError(f"Dataset has no audio/ directory: {dataset_root}")

    selected_episode_indices = episode_indices or _episode_indices(dataset_root)
    if not selected_episode_indices:
        raise RuntimeError(f"No episodes found under {dataset_root}")

    model, timestamp_fn, model_info = _load_silero()

    if print_summary:
        print("=" * 70)
        print("Offline Silero-VAD preprocessing")
        print("=" * 70)
        print(f"Dataset: {dataset_root}")
        print(f"Episodes: {len(selected_episode_indices)}")
        print(f"Save clips: {args.save_clips}")
        print(f"Model loader: {model_info.get('model_loader')}")
        if model_info.get("model_loader") == "torch.jit.load":
            print("[Warning] Using direct Silero JIT fallback.")
        print("=" * 70)

    processed = 0
    skipped = 0
    missing = 0
    total_segments = 0
    results: list[dict[str, Any]] = []
    for episode_index in selected_episode_indices:
        result = process_episode(
            dataset_root=dataset_root,
            episode_index=int(episode_index),
            model=model,
            timestamp_fn=timestamp_fn,
            model_info=model_info,
            args=args,
        )
        results.append(result)
        status = result["status"]
        if status == "processed":
            processed += 1
            total_segments += int(result["num_segments"])
            if print_summary:
                print(
                    f"[VAD] episode_{episode_index:06d}: "
                    f"{result['num_segments']} segment(s)"
                )
        elif status == "skipped":
            skipped += 1
            if print_summary:
                print(
                    f"[VAD] episode_{episode_index:06d}: skipped ({result['reason']})"
                )
        else:
            missing += 1
            if print_summary:
                print(f"[VAD] episode_{episode_index:06d}: {status}")

    summary = {
        "dataset_path": str(dataset_root),
        "episodes": len(selected_episode_indices),
        "processed": processed,
        "skipped": skipped,
        "missing": missing,
        "vad_segments": total_segments,
        "results": results,
    }

    window_summary = None
    if bool(args.instruction_audio_window):
        window_summary = refresh_dataset_instruction_audio_windows(
            dataset_root,
            episode_indices=[int(index) for index in selected_episode_indices],
            pre_margin_sec=float(args.instruction_pre_margin_sec),
            post_margin_sec=float(args.instruction_post_margin_sec),
            overwrite=True,
            dry_run=False,
        )
        summary["instruction_audio_window"] = window_summary

    if print_summary:
        print("=" * 70)
        print(
            "Done: "
            f"processed={processed}, skipped={skipped}, missing={missing}, "
            f"vad_segments={total_segments}"
        )
        if window_summary is not None:
            print(
                "Instruction windows: "
                f"updated={window_summary['updated']}, "
                f"vad={window_summary['vad_windows']}, "
                f"imputed={window_summary['imputed_windows']}, "
                f"pending={window_summary['pending_windows']}, "
                f"reference={window_summary['reference_window_count']}"
            )

    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    compute_vad_for_dataset(
        args.dataset_path,
        episode_indices=args.episode_index,
        overwrite=bool(args.overwrite),
        save_clips=bool(args.save_clips),
        clips_dir=args.clips_dir,
        threshold=float(args.threshold),
        neg_threshold=args.neg_threshold,
        min_speech_duration_ms=int(args.min_speech_duration_ms),
        max_speech_duration_s=float(args.max_speech_duration_s),
        min_silence_duration_ms=int(args.min_silence_duration_ms),
        speech_pad_ms=int(args.speech_pad_ms),
        instruction_audio_window=bool(args.instruction_audio_window),
        instruction_pre_margin_sec=float(args.instruction_pre_margin_sec),
        instruction_post_margin_sec=float(args.instruction_post_margin_sec),
        print_summary=True,
    )


if __name__ == "__main__":
    main()
