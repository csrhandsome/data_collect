"""Precompute static instruction-audio encoder features per episode.

Supports Franka realworld ASR sidecars under ``data/dataset/...`` and pluggable
encoders from the sibling openpi checkout (``openpi.shared.audio_tools``):

  - ``qwen``   → ``<dataset_root>/qwen_instruction_features/episode_XXXXXX.npy``
  - ``usad``   → ``<dataset_root>/usad_instruction_features/episode_XXXXXX.npy``
  - ``whisper``→ ``<dataset_root>/whisper_instruction_features/episode_XXXXXX.npy``

Examples:
    # Qwen dry-run
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \\
      $OPENPI_ROOT/.venv/bin/python scripts/precompute_instruction_features.py \\
      --encoder qwen --dataset-root data/dataset/franka_lerobot_7_6_audio \\
      --dry-run --limit 1 --workers 1 --threads-per-worker 4

    # USAD dry-run (MIT-SLS/USAD2-Large)
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \\
      $OPENPI_ROOT/.venv/bin/python scripts/precompute_instruction_features.py \\
      --encoder usad --dataset-root data/dataset/franka_lerobot_7_6_audio \\
      --dry-run --limit 1 --workers 1 --threads-per-worker 4

    # Full USAD precompute
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \\
      $OPENPI_ROOT/.venv/bin/python scripts/precompute_instruction_features.py \\
      --encoder usad --dataset-root data/dataset/franka_lerobot_7_6_audio \\
      --workers 4 --threads-per-worker 4
"""

from __future__ import annotations

import argparse
from concurrent import futures
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable

import numpy as np
import tqdm

# Prefer China HF mirror unless the caller already set one (matches other ASR scripts).
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
os.environ.setdefault("HF_HUB_ENDPOINT", DEFAULT_HF_ENDPOINT)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "dataset" / "franka_lerobot_7_6_audio"
_OPENPI_CANDIDATES = (
    Path(os.environ["OPENPI_ROOT"]) if os.environ.get("OPENPI_ROOT") else None,
    Path("/home/three/DataFromRoot/codes/grade_two/github/robot/openpi"),
    Path("/data/codes/grade_two/github/robot/openpi"),
)
_EPISODE_PATTERN = re.compile(r"episode_(\d+)\.sync\.json$")
_WORKER_CONFIG: dict[str, object] = {}
SUPPORTED_ENCODERS = ("qwen", "usad", "whisper")


def _default_openpi_root() -> Path:
    for candidate in _OPENPI_CANDIDATES:
        if candidate is None:
            continue
        if (candidate / "src" / "openpi" / "shared" / "audio_tools.py").is_file():
            return candidate
    return Path("/data/codes/grade_two/github/robot/openpi")


def _ensure_openpi_on_path(openpi_root: Path) -> None:
    src = (openpi_root / "src").resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"openpi src not found at {src}")
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _feature_cache_dir(dataset_root: str | Path, encoder: str) -> Path:
    from openpi.shared import audio_tools

    return audio_tools.instruction_feature_cache_dir(dataset_root, encoder)


def _default_model_name(encoder: str) -> str:
    if encoder == "qwen":
        return "Qwen/Qwen3-ASR-0.6B"
    if encoder == "usad":
        return "MIT-SLS/USAD2-Large"
    if encoder == "whisper":
        return "openai/whisper-tiny"
    raise ValueError(f"Unsupported encoder {encoder!r}")


def _resolve_encoder_geometry(
    *,
    encoder: str,
    model_name: str,
    sample_rate: int,
    instruction_num_samples: int,
) -> tuple[int, int]:
    from openpi.shared import audio_tools

    if encoder == "qwen":
        num_frames = audio_tools.qwen_asr_num_frames_for_duration(
            instruction_num_samples,
            sample_rate=sample_rate,
        )
        dim = audio_tools.infer_qwen_asr_hidden_size(model_name)
        return num_frames, dim
    if encoder == "usad":
        num_frames = audio_tools.usad_num_frames_for_duration(
            instruction_num_samples,
            sample_rate=sample_rate,
            model_name=model_name,
        )
        dim = audio_tools.infer_usad_hidden_size(model_name)
        return num_frames, dim
    if encoder == "whisper":
        num_frames = audio_tools.whisper_num_frames_for_duration(
            instruction_num_samples,
            sample_rate=sample_rate,
        )
        dim = audio_tools.infer_whisper_hidden_size(model_name)
        return num_frames, dim
    raise ValueError(f"Unsupported encoder {encoder!r}")


def _load_encoder_components(encoder: str, model_name: str, device: str) -> None:
    from openpi.shared import audio_tools

    if encoder == "qwen":
        audio_tools._load_qwen_asr_components(model_name, device)  # noqa: SLF001
    elif encoder == "usad":
        audio_tools._load_usad_components(model_name, device)  # noqa: SLF001
    elif encoder == "whisper":
        audio_tools._load_whisper_components(model_name, device)  # noqa: SLF001
    else:
        raise ValueError(f"Unsupported encoder {encoder!r}")


def _compute_encoder_sequence(
    encoder: str,
    audio: np.ndarray,
    *,
    sample_rate: int,
    model_name: str,
    device: str,
) -> np.ndarray:
    from openpi.shared import audio_tools

    compute: Callable[..., np.ndarray]
    if encoder == "qwen":
        compute = audio_tools.compute_qwen_asr_audio_sequence
    elif encoder == "usad":
        compute = audio_tools.compute_usad_audio_sequence
    elif encoder == "whisper":
        compute = audio_tools.compute_whisper_audio_sequence
    else:
        raise ValueError(f"Unsupported encoder {encoder!r}")
    return compute(
        audio,
        sample_rate=sample_rate,
        model_name=model_name,
        device=device,
    )


def _instruction_frame_index(
    audio_sidecar_dir: Path,
    episode_index: int,
    fallback_fps: float,
) -> int | None:
    """Pick a frame where static instruction audio is active.

    Compatibility:
      - Franka ASR: ``instruction_audio_window`` usually starts ~9s in; use mid-window.
      - Missing/invalid window: fall back to frame 0 and let extraction decide validity.
    """
    sync_path = audio_sidecar_dir / "audio" / f"episode_{episode_index:06d}.sync.json"
    if not sync_path.is_file():
        return None
    sync_data = json.loads(sync_path.read_text())
    fps = float(sync_data.get("control_frequency", fallback_fps))
    video_frames = sync_data.get("video_frames")
    max_frame = max(0, int(video_frames) - 1) if video_frames is not None else None

    window = sync_data.get("instruction_audio_window")
    if isinstance(window, dict):
        if not bool(window.get("audio_valid", True)):
            return None
        start_sec = window.get("start_sec")
        end_sec = window.get("end_sec")
        if start_sec is not None and end_sec is not None and float(end_sec) >= float(start_sec):
            mid_sec = 0.5 * (float(start_sec) + float(end_sec))
            frame_index = max(0, int(mid_sec * fps))
            if max_frame is not None:
                frame_index = min(frame_index, max_frame)
            return frame_index

    return 0


def _init_worker(
    encoder: str,
    model_name: str,
    device: str,
    worker_count: int,
    threads_per_worker: int,
    audio_sidecar_dir: str,
    output_dir: str,
    sample_rate: int,
    instruction_num_samples: int,
    expected_num_frames: int,
    expected_dim: int,
    output_dtype: str,
    overwrite: int,
    fallback_fps: float,
    dry_run: int,
    openpi_root: str,
) -> None:
    import torch

    _ensure_openpi_on_path(Path(openpi_root))

    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        allowed_cpus = sorted(os.sched_getaffinity(0))
        physical_cpus = [cpu for cpu in allowed_cpus if cpu < len(allowed_cpus) // 2]
        candidate_cpus = physical_cpus if len(physical_cpus) >= worker_count * threads_per_worker else allowed_cpus
        process_identity = multiprocessing.current_process()._identity  # noqa: SLF001
        worker_index = ((process_identity[0] - 1) if process_identity else 0) % worker_count
        start = worker_index * threads_per_worker
        worker_cpus = candidate_cpus[start : start + threads_per_worker]
        if len(worker_cpus) == threads_per_worker:
            os.sched_setaffinity(0, worker_cpus)

    torch.set_num_threads(threads_per_worker)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)

    _WORKER_CONFIG.update(
        {
            "encoder": encoder,
            "model_name": model_name,
            "device": device,
            "audio_sidecar_dir": audio_sidecar_dir,
            "output_dir": output_dir,
            "sample_rate": sample_rate,
            "instruction_num_samples": instruction_num_samples,
            "expected_num_frames": expected_num_frames,
            "expected_dim": expected_dim,
            "output_dtype": output_dtype,
            "overwrite": bool(overwrite),
            "fallback_fps": float(fallback_fps),
            "dry_run": bool(dry_run),
            "openpi_root": openpi_root,
        }
    )

    _load_encoder_components(encoder, model_name, device)


def _precompute_episode(episode_index: int) -> tuple[int, str, float, tuple[int, ...] | None]:
    from openpi.shared import audio_tools

    start = time.perf_counter()
    output_dir = Path(str(_WORKER_CONFIG["output_dir"]))
    output_path = output_dir / f"episode_{episode_index:06d}.npy"
    dry_run = bool(_WORKER_CONFIG["dry_run"])
    if output_path.is_file() and not bool(_WORKER_CONFIG["overwrite"]) and not dry_run:
        return episode_index, "skipped", time.perf_counter() - start, None

    encoder = str(_WORKER_CONFIG["encoder"])
    sample_rate = int(_WORKER_CONFIG["sample_rate"])
    instruction_num_samples = int(_WORKER_CONFIG["instruction_num_samples"])
    expected_num_frames = int(_WORKER_CONFIG["expected_num_frames"])
    expected_dim = int(_WORKER_CONFIG["expected_dim"])
    fallback_fps = float(_WORKER_CONFIG["fallback_fps"])
    audio_sidecar_dir = Path(str(_WORKER_CONFIG["audio_sidecar_dir"]))

    frame_index = _instruction_frame_index(audio_sidecar_dir, episode_index, fallback_fps)
    if frame_index is None:
        return episode_index, "invalid", time.perf_counter() - start, None

    audio, valid = audio_tools.extract_episode_instruction_audio(
        dataset_root=str(audio_sidecar_dir),
        episode_index=episode_index,
        frame_index=frame_index,
        window_num_samples=instruction_num_samples,
        target_sample_rate=sample_rate,
        fallback_fps=fallback_fps,
    )
    if not valid:
        return episode_index, "invalid", time.perf_counter() - start, None

    feature = _compute_encoder_sequence(
        encoder,
        audio,
        sample_rate=sample_rate,
        model_name=str(_WORKER_CONFIG["model_name"]),
        device=str(_WORKER_CONFIG["device"]),
    )
    feature = audio_tools.fit_audio_feature_sequence(feature, expected_num_frames)
    if feature.shape != (expected_num_frames, expected_dim):
        raise ValueError(
            f"Episode {episode_index} feature shape mismatch: "
            f"expected {(expected_num_frames, expected_dim)}, got {feature.shape}."
        )

    feature = np.asarray(feature, dtype=np.dtype(str(_WORKER_CONFIG["output_dtype"])))
    if dry_run:
        return episode_index, "dry_run", time.perf_counter() - start, tuple(int(x) for x in feature.shape)

    temporary_path = output_dir / f".episode_{episode_index:06d}.{os.getpid()}.tmp"
    with temporary_path.open("wb") as file:
        np.save(file, feature, allow_pickle=False)
    os.replace(temporary_path, output_path)
    return episode_index, "written", time.perf_counter() - start, tuple(int(x) for x in feature.shape)


def _episode_indices(audio_sidecar_dir: Path) -> list[int]:
    indices = []
    for sync_path in sorted((audio_sidecar_dir / "audio").glob("episode_*.sync.json")):
        match = _EPISODE_PATTERN.search(sync_path.name)
        if match is not None:
            indices.append(int(match.group(1)))
    if not indices:
        raise FileNotFoundError(f"No episode sync files found under {audio_sidecar_dir / 'audio'}")
    return indices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute per-episode instruction-audio encoder features (qwen/usad/whisper)."
    )
    parser.add_argument(
        "--encoder",
        choices=SUPPORTED_ENCODERS,
        default="qwen",
        help="Audio encoder backend. Output goes to <dataset>/<encoder>_instruction_features/.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="LeRobot dataset root containing audio/episode_*.sync.json.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=_default_openpi_root(),
        help="openpi checkout used for audio_tools (default: DataFromRoot or /data openpi).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Encode episodes but do not write .npy / metadata.json.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Encoder checkpoint id. Defaults depend on --encoder.",
    )
    parser.add_argument("--qwen-model-name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--usad-model-name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--whisper-model-name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--instruction-num-samples",
        type=int,
        default=32_000,
        help="Matches openpi enable_*_encoder default (2s @ 16kHz).",
    )
    parser.add_argument("--fallback-fps", type=float, default=20.0)
    return parser.parse_args()


def _resolve_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return str(args.model_name)
    legacy = {
        "qwen": args.qwen_model_name,
        "usad": args.usad_model_name,
        "whisper": args.whisper_model_name,
    }.get(args.encoder)
    if legacy:
        return str(legacy)
    return _default_model_name(args.encoder)


def main() -> None:
    args = _parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.threads_per_worker <= 0:
        raise ValueError("--threads-per-worker must be > 0")

    encoder = str(args.encoder)
    model_name = _resolve_model_name(args)
    openpi_root = args.openpi_root.expanduser().resolve()
    _ensure_openpi_on_path(openpi_root)

    if hasattr(os, "sched_setaffinity"):
        with contextlib.suppress(OSError):
            os.sched_setaffinity(0, range(os.cpu_count() or 1))

    audio_sidecar_dir = args.dataset_root.expanduser().resolve()
    if not (audio_sidecar_dir / "audio").is_dir():
        raise FileNotFoundError(f"Missing audio/ under dataset root: {audio_sidecar_dir}")

    output_dir = args.output_dir or _feature_cache_dir(audio_sidecar_dir, encoder)
    output_dir = output_dir.expanduser().resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    episode_indices = _episode_indices(audio_sidecar_dir)
    if args.limit is not None:
        episode_indices = episode_indices[: args.limit]

    sample_rate = int(args.sample_rate)
    instruction_num_samples = int(args.instruction_num_samples)
    expected_num_frames, expected_dim = _resolve_encoder_geometry(
        encoder=encoder,
        model_name=model_name,
        sample_rate=sample_rate,
        instruction_num_samples=instruction_num_samples,
    )
    fallback_fps = float(args.fallback_fps)

    for env_name, value in {
        "OMP_NUM_THREADS": args.threads_per_worker,
        "MKL_NUM_THREADS": args.threads_per_worker,
        "OPENBLAS_NUM_THREADS": 1,
        "NUMEXPR_NUM_THREADS": min(args.threads_per_worker, 4),
        "TOKENIZERS_PARALLELISM": "false",
        "MALLOC_ARENA_MAX": 4,
    }.items():
        os.environ[env_name] = str(value)

    metadata = {
        "format_version": 1,
        "encoder": encoder,
        "dataset_root": str(audio_sidecar_dir),
        "audio_sidecar_dir": str(audio_sidecar_dir),
        "model_name": model_name,
        "sample_rate": sample_rate,
        "instruction_num_samples": instruction_num_samples,
        "feature_shape": [expected_num_frames, expected_dim],
        "dtype": args.dtype,
        "fallback_fps": fallback_fps,
        "episodes": len(episode_indices),
        "dry_run": bool(args.dry_run),
        "openpi_root": str(openpi_root),
    }
    if not args.dry_run:
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    else:
        print(f"[dry-run] metadata={json.dumps(metadata)}")

    written = 0
    skipped = 0
    invalid = 0
    dry_run_ok = 0
    context = multiprocessing.get_context("spawn")
    with futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(
            encoder,
            model_name,
            args.device,
            args.workers,
            args.threads_per_worker,
            str(audio_sidecar_dir),
            str(output_dir),
            sample_rate,
            instruction_num_samples,
            expected_num_frames,
            expected_dim,
            args.dtype,
            int(args.overwrite),
            fallback_fps,
            int(args.dry_run),
            str(openpi_root),
        ),
    ) as executor:
        pending = [executor.submit(_precompute_episode, episode_index) for episode_index in episode_indices]
        desc = f"Precomputing {encoder} features"
        with tqdm.tqdm(total=len(pending), desc=desc, dynamic_ncols=True) as progress:
            for future in futures.as_completed(pending):
                episode_index, status, elapsed, shape = future.result()
                written += status == "written"
                skipped += status == "skipped"
                invalid += status == "invalid"
                dry_run_ok += status == "dry_run"
                progress.set_postfix(
                    episode=episode_index,
                    status=status,
                    shape=shape,
                    seconds=f"{elapsed:.2f}",
                    written=written,
                    skipped=skipped,
                    invalid=invalid,
                    dry_run=dry_run_ok,
                )
                progress.update()

    print(
        f"Done. encoder={encoder} output_dir={output_dir} written={written} "
        f"skipped={skipped} invalid={invalid} dry_run={dry_run_ok}"
    )


if __name__ == "__main__":
    main()
