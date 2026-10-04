"""Convert an audio LeRobot dataset into a Qwen3-ASR-prompt dataset.

LEGACY: 仅保留历史代码。当前仓库环境不支持 ASR 或 ASR 模型特征提取，
不安装相关依赖，也不提供兼容补丁。使用需要自行配置独立的 legacy 环境。

This is intentionally non-destructive: by default it copies the input dataset
to a new directory whose name is suffixed with ``_qwen_asr`` and rewrites only
the copy.

Default usage:
  uv run data_analysis/convert_audio_dataset_to_asr_qwen.py --overwrite-output

Explicit usage:
uv run data_analysis/convert_audio_dataset_to_asr_qwen.py \
    --input-dir data/dataset/franka_lerobot_6_19_audio \
    --output-dir data/dataset/franka_lerobot_6_19_audio_asr \
    --overwrite-output \
    --asr-model Qwen/Qwen3-ASR-0.6B

If ``qwen_asr`` is not installed but the local Qwen3-ASR clone exists, this
script falls back to importing it from ``QWEN3_ASR_REPO`` or the default clone
path below.
"""

from __future__ import annotations

import argparse
import functools
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
os.environ.setdefault("HF_HUB_ENDPOINT", DEFAULT_HF_ENDPOINT)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_QWEN3_ASR_REPO = PROJECT_ROOT / "third_party" / "Qwen3-ASR"

from control.util.audio_util import read_wav_pcm  # noqa: E402
from data_analysis.convert_audio_dataset_to_asr import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    _fallback_task,
    _is_lerobot_dataset,
    _normalize_prompt,
    _read_json,
    _resolve_dataset_relative_path,
    _resolve_existing_path,
    _rewrite_metadata_and_parquet,
    _write_json,
)


@dataclass(frozen=True)
class QwenTranscript:
    text: str
    language: str


def _default_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(f"{input_dir.name}_qwen_asr")


def _maybe_add_qwen3_asr_repo_to_path() -> None:
    candidates = []
    env_repo = os.environ.get("QWEN3_ASR_REPO")
    if env_repo:
        candidates.append(Path(env_repo))
    candidates.append(DEFAULT_QWEN3_ASR_REPO)

    for repo_dir in candidates:
        package_init = repo_dir / "qwen_asr" / "__init__.py"
        if package_init.is_file():
            resolved = str(repo_dir.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            return


def _import_qwen3_asr_model():
    try:
        from qwen_asr import Qwen3ASRModel

        return Qwen3ASRModel
    except ModuleNotFoundError as exc:
        if exc.name != "qwen_asr":
            raise

    _maybe_add_qwen3_asr_repo_to_path()
    try:
        from qwen_asr import Qwen3ASRModel

        return Qwen3ASRModel
    except ModuleNotFoundError as exc:
        if exc.name == "qwen_asr":
            raise RuntimeError(
                "Qwen3-ASR requires `qwen-asr`. Install it, or set "
                "QWEN3_ASR_REPO to a local Qwen3-ASR checkout."
            ) from exc
        raise


def _resolve_qwen_device_map(device: str) -> str:
    if device != "auto":
        return device

    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_torch_dtype(dtype: str, resolved_device: str):
    import torch

    normalized = dtype.strip().lower()
    if normalized == "auto":
        normalized = "bfloat16" if resolved_device.startswith("cuda") else "float32"

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in dtype_by_name:
        raise ValueError(
            f"Unsupported --asr-dtype {dtype!r}; use auto, bfloat16, float16, or float32."
        )
    return dtype_by_name[normalized], normalized


@functools.lru_cache(maxsize=4)
def _load_qwen3_asr_model(
    model_name: str,
    device: str,
    dtype: str,
    max_inference_batch_size: int,
    max_new_tokens: int,
    attn_implementation: str | None,
):
    Qwen3ASRModel = _import_qwen3_asr_model()
    resolved_device = _resolve_qwen_device_map(device)
    torch_dtype, resolved_dtype = _resolve_torch_dtype(dtype, resolved_device)

    kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "device_map": resolved_device,
        "max_inference_batch_size": int(max_inference_batch_size),
        "max_new_tokens": int(max_new_tokens),
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = Qwen3ASRModel.from_pretrained(model_name, **kwargs)
    return model, resolved_device, resolved_dtype


def _normalize_qwen_language(language: str | None) -> str | None:
    if language is None:
        return None
    value = str(language).strip()
    if not value or value.lower() in {"auto", "none", "null"}:
        return None

    aliases = {
        "zh": "Chinese",
        "cn": "Chinese",
        "chs": "Chinese",
        "chinese": "Chinese",
        "mandarin": "Chinese",
        "en": "English",
        "eng": "English",
        "english": "English",
        "yue": "Cantonese",
        "cantonese": "Cantonese",
    }
    return aliases.get(value.lower(), value[:1].upper() + value[1:].lower())


def _extract_result_field(result: object, field: str) -> str:
    if isinstance(result, dict):
        return str(result.get(field) or "")
    return str(getattr(result, field, "") or "")


def _transcribe_wav(
    path: Path,
    *,
    model_name: str,
    device: str,
    language: str | None,
    dtype: str,
    max_inference_batch_size: int,
    max_new_tokens: int,
    attn_implementation: str | None,
    context: str,
) -> QwenTranscript:
    audio, sample_rate = read_wav_pcm(path, require_nonempty=True)
    model, _, _ = _load_qwen3_asr_model(
        model_name,
        device,
        dtype,
        int(max_inference_batch_size),
        int(max_new_tokens),
        attn_implementation,
    )
    results = model.transcribe(
        audio=(audio, int(sample_rate)),
        context=context,
        language=_normalize_qwen_language(language),
        return_time_stamps=False,
    )
    if not results:
        return QwenTranscript(text="", language="")
    result = results[0]
    return QwenTranscript(
        text=_extract_result_field(result, "text").strip(),
        language=_extract_result_field(result, "language").strip(),
    )


def _collect_episode_prompt(
    *,
    dataset_dir: Path,
    episode_row: dict[str, Any],
    model_name: str,
    device: str,
    language: str | None,
    dtype: str,
    max_inference_batch_size: int,
    max_new_tokens: int,
    attn_implementation: str | None,
    context: str,
    transcribe_full_audio_when_no_vad: bool,
    dry_run: bool,
) -> tuple[str, dict[str, Any]]:
    episode_index = int(episode_row["episode_index"])
    stem = f"episode_{episode_index:06d}"
    sync_path = dataset_dir / "audio" / f"{stem}.sync.json"
    sync_data = _read_json(sync_path) if sync_path.is_file() else {}
    fallback_task = _fallback_task(episode_row, sync_data)

    vad_segments = sync_data.get("vad_segments")
    if not isinstance(vad_segments, list):
        vad_segments = []

    transcript_parts: list[str] = []
    detected_languages: list[str] = []
    errors: list[dict[str, str]] = []
    source = "vad_segments"

    for segment in vad_segments:
        if not isinstance(segment, dict):
            continue
        wav_path = _resolve_dataset_relative_path(
            dataset_dir, segment.get("audio_wav_path")
        )
        if wav_path is None or not wav_path.is_file():
            continue
        try:
            result = _transcribe_wav(
                wav_path,
                model_name=model_name,
                device=device,
                language=language,
                dtype=dtype,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=max_new_tokens,
                attn_implementation=attn_implementation,
                context=context,
            )
            transcript = _normalize_prompt(result.text)
            if transcript:
                transcript_parts.append(transcript)
            if result.language:
                detected_languages.append(result.language)
            if not dry_run:
                segment["asr_transcript"] = transcript
                segment["asr_language"] = result.language or None
                segment["asr_error"] = None
        except Exception as exc:
            errors.append({"source": str(wav_path), "error": str(exc)})
            if not dry_run:
                segment["asr_error"] = str(exc)

    if not transcript_parts and transcribe_full_audio_when_no_vad and not vad_segments:
        wav_path = _resolve_dataset_relative_path(
            dataset_dir, sync_data.get("audio_path")
        )
        if wav_path is None:
            wav_path = dataset_dir / "audio" / f"{stem}.wav"
        if wav_path.is_file():
            source = "full_episode_audio"
            try:
                result = _transcribe_wav(
                    wav_path,
                    model_name=model_name,
                    device=device,
                    language=language,
                    dtype=dtype,
                    max_inference_batch_size=max_inference_batch_size,
                    max_new_tokens=max_new_tokens,
                    attn_implementation=attn_implementation,
                    context=context,
                )
                transcript = _normalize_prompt(result.text)
                if transcript:
                    transcript_parts.append(transcript)
                if result.language:
                    detected_languages.append(result.language)
            except Exception as exc:
                errors.append({"source": str(wav_path), "error": str(exc)})

    transcript_text = _normalize_prompt(" ".join(transcript_parts))
    prompt = transcript_text or fallback_task
    used_asr = bool(transcript_text)
    forced_language = _normalize_qwen_language(language)

    metadata = {
        "enabled": True,
        "backend": "qwen3-asr",
        "model": model_name,
        "device": device,
        "language": forced_language,
        "dtype": dtype,
        "max_inference_batch_size": int(max_inference_batch_size),
        "max_new_tokens": int(max_new_tokens),
        "attn_implementation": attn_implementation,
        "context": context,
        "source": source,
        "transcript": transcript_text,
        "detected_languages": detected_languages,
        "fallback_task": fallback_task,
        "used_for_task": used_asr,
        "num_vad_segments": len(vad_segments),
        "errors": errors,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        sync_data["episode_index"] = episode_index
        sync_data["task"] = prompt
        sync_data["fallback_task"] = fallback_task
        sync_data["asr_metadata"] = metadata
        sync_data["vad_segments"] = vad_segments
        _write_json(sync_path, sync_data)

    return prompt, metadata


def convert_dataset(args: argparse.Namespace) -> int:
    if args.qwen3_asr_repo is not None:
        os.environ["QWEN3_ASR_REPO"] = str(args.qwen3_asr_repo)

    input_dir = _resolve_existing_path(args.input_dir)
    if not _is_lerobot_dataset(input_dir):
        print(f"[ERROR] Not a LeRobot dataset: {input_dir}", file=sys.stderr)
        return 2

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else _default_output_dir(input_dir)
    )

    if input_dir == output_dir and not args.dry_run:
        print(
            "[ERROR] Refusing to overwrite the input dataset in place.", file=sys.stderr
        )
        return 2

    print(f"Input dataset:       {input_dir}")
    print(f"Output dataset:      {output_dir}")
    print("ASR backend:         Qwen3-ASR")
    print(f"ASR model:           {args.asr_model}")
    print(f"ASR device:          {args.asr_device}")
    print(f"ASR dtype:           {args.asr_dtype}")
    print(f"ASR language:        {args.asr_language or 'auto'}")
    print(f"ASR max new tokens:  {args.asr_max_new_tokens}")

    if not args.dry_run:
        if output_dir.exists():
            if not args.overwrite_output:
                print(
                    "[ERROR] Output directory already exists. "
                    "Use --overwrite-output to replace it.",
                    file=sys.stderr,
                )
                return 2
            shutil.rmtree(output_dir)
        print("[Copy] Copying dataset...")
        shutil.copytree(input_dir, output_dir)
        work_dir = output_dir
    else:
        print("[Dry-run] No files will be written.")
        work_dir = input_dir

    episodes_path = work_dir / "meta" / "episodes.jsonl"
    from data_analysis.dataset_io import episode_rows as load_episode_rows
    episode_rows = load_episode_rows(work_dir)
    if not episode_rows:
        print(f"[ERROR] No episodes found in {episodes_path}", file=sys.stderr)
        return 2

    prompts_by_episode: dict[int, str] = {}
    asr_used = 0
    errors = 0

    for row in tqdm(episode_rows, desc="Transcribing VAD clips with Qwen3-ASR"):
        episode_index = int(row["episode_index"])
        prompt, metadata = _collect_episode_prompt(
            dataset_dir=work_dir,
            episode_row=row,
            model_name=args.asr_model,
            device=args.asr_device,
            language=args.asr_language,
            dtype=args.asr_dtype,
            max_inference_batch_size=args.asr_max_inference_batch_size,
            max_new_tokens=args.asr_max_new_tokens,
            attn_implementation=args.asr_attn_implementation,
            context=args.asr_context,
            transcribe_full_audio_when_no_vad=args.transcribe_full_audio_when_no_vad,
            dry_run=args.dry_run,
        )
        prompts_by_episode[episode_index] = prompt
        asr_used += int(bool(metadata["used_for_task"]))
        errors += len(metadata["errors"])

    rewrite_summary = _rewrite_metadata_and_parquet(
        dataset_dir=work_dir,
        episode_rows=episode_rows,
        prompts_by_episode=prompts_by_episode,
        dry_run=args.dry_run,
    )

    print("=" * 70)
    print("Qwen3-ASR conversion summary")
    print(f"Episodes:        {len(episode_rows)}")
    print(f"ASR-used prompts:{asr_used}")
    print(f"Fallback prompts:{len(episode_rows) - asr_used}")
    print(f"Unique tasks:    {rewrite_summary['num_tasks']}")
    print(f"ASR errors:      {errors}")
    print(f"Output dataset:  {output_dir if not args.dry_run else '(dry-run)'}")
    print("=" * 70)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy a LeRobot audio dataset and replace prompts with Qwen3-ASR text."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Input LeRobot dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input_dir>_qwen_asr.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete and recreate --output-dir if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run ASR and print a summary without copying or modifying files.",
    )
    parser.add_argument(
        "--qwen3-asr-repo",
        type=Path,
        default=None,
        help=(
            "Optional local Qwen3-ASR checkout. Used only if `qwen_asr` is not "
            "already importable."
        ),
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
        help="Qwen3-ASR model name or local model path.",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="auto",
        help="Qwen3-ASR device_map, e.g. auto, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--asr-dtype",
        type=str,
        default="auto",
        help="Qwen3-ASR torch dtype: auto, bfloat16, float16, or float32.",
    )
    parser.add_argument(
        "--asr-language",
        type=str,
        default="Chinese",
        help="Optional Qwen language, e.g. Chinese, English, Cantonese, or auto.",
    )
    parser.add_argument(
        "--asr-context",
        type=str,
        default="",
        help="Optional context text passed to Qwen3-ASR.",
    )
    parser.add_argument(
        "--asr-max-inference-batch-size",
        type=int,
        default=32,
        help="Qwen3-ASR inference batch size limit. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--asr-max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens generated by Qwen3-ASR.",
    )
    parser.add_argument(
        "--asr-attn-implementation",
        type=str,
        default=None,
        help="Optional attention implementation, e.g. flash_attention_2.",
    )
    parser.add_argument(
        "--transcribe-full-audio-when-no-vad",
        action="store_true",
        help="If an episode has no VAD segments, transcribe the full episode WAV.",
    )
    return parser


def main() -> None:
    raise SystemExit(convert_dataset(build_parser().parse_args()))


if __name__ == "__main__":
    main()
