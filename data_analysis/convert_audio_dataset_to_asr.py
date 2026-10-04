"""Convert an audio LeRobot dataset into an ASR-prompt dataset.

This is intentionally non-destructive: by default it copies the input dataset
to a new directory whose name is suffixed with ``_asr`` and rewrites only the
copy.

Default usage:
  uv run data_analysis/convert_audio_dataset_to_asr.py --overwrite-output

Explicit usage:
  uv run data_analysis/convert_audio_dataset_to_asr.py \
    --input-dir data/dataset/franka_lerobot_4_9_audio \
    --overwrite-output
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
os.environ.setdefault("HF_HUB_ENDPOINT", DEFAULT_HF_ENDPOINT)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.util.audio_util import read_wav_pcm, transcribe_whisper_asr  # noqa: E402

DEFAULT_INPUT_DIR = Path("data/dataset/franka_lerobot_4_9_audio")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if text:
        text += "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _resolve_existing_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path

    project_path = (PROJECT_ROOT / path).resolve()
    if project_path.exists():
        return project_path

    return cwd_path


def _default_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(f"{input_dir.name}_asr")


def _is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _episode_parquet_path(
    info: dict[str, Any], dataset_dir: Path, episode_index: int
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = int(episode_index) // chunks_size
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    rel = template.format(
        episode_chunk=episode_chunk,
        episode_index=int(episode_index),
    )
    return dataset_dir / rel


def _resolve_dataset_relative_path(dataset_dir: Path, raw_path: object) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return dataset_dir / path


def _normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _transcribe_wav(
    path: Path,
    *,
    model_name: str,
    device: str,
    language: str | None,
    max_new_tokens: int,
) -> str:
    audio, sample_rate = read_wav_pcm(path, require_nonempty=True)
    return transcribe_whisper_asr(
        audio,
        sample_rate=int(sample_rate),
        model_name=model_name,
        device=device,
        language=language,
        max_new_tokens=int(max_new_tokens),
    )


def _fallback_task(episode_row: dict[str, Any], sync_data: dict[str, Any]) -> str:
    tasks = episode_row.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    task = sync_data.get("task")
    if task:
        return str(task)
    return ""


def _collect_episode_prompt(
    *,
    dataset_dir: Path,
    episode_row: dict[str, Any],
    model_name: str,
    device: str,
    language: str | None,
    max_new_tokens: int,
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
            transcript = _normalize_prompt(
                _transcribe_wav(
                    wav_path,
                    model_name=model_name,
                    device=device,
                    language=language,
                    max_new_tokens=max_new_tokens,
                )
            )
            if transcript:
                transcript_parts.append(transcript)
            if not dry_run:
                segment["asr_transcript"] = transcript
                segment["asr_error"] = None
        except Exception as exc:
            errors.append(
                {
                    "source": str(wav_path),
                    "error": str(exc),
                }
            )
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
                transcript = _normalize_prompt(
                    _transcribe_wav(
                        wav_path,
                        model_name=model_name,
                        device=device,
                        language=language,
                        max_new_tokens=max_new_tokens,
                    )
                )
                if transcript:
                    transcript_parts.append(transcript)
            except Exception as exc:
                errors.append(
                    {
                        "source": str(wav_path),
                        "error": str(exc),
                    }
                )

    transcript_text = _normalize_prompt(" ".join(transcript_parts))
    prompt = transcript_text or fallback_task
    used_asr = bool(transcript_text)

    metadata = {
        "enabled": True,
        "model": model_name,
        "device": device,
        "language": language,
        "source": source,
        "transcript": transcript_text,
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


def _replace_column(table: pa.Table, column_name: str, values: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(column_name)
    if idx < 0:
        raise RuntimeError(f"parquet missing column: {column_name}")
    return table.set_column(idx, table.schema.field(idx), values)


def _rewrite_parquet_task_index(
    *,
    path: Path,
    task_index: int,
    expected_length: int,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing parquet file: {path}")

    table = pq.read_table(path)
    if table.num_rows != expected_length:
        raise RuntimeError(
            f"Parquet row count does not match episodes.jsonl length: {path} "
            f"rows={table.num_rows}, length={expected_length}"
        )

    field = table.schema.field("task_index")
    values = pa.array([int(task_index)] * table.num_rows, type=field.type)
    table = _replace_column(table, "task_index", values)

    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def _patch_task_index_stats(
    row: dict[str, Any], *, task_index: int, length: int
) -> dict[str, Any]:
    row = dict(row)
    stats = row.get("stats")
    if not isinstance(stats, dict):
        return row

    task_stats = stats.get("task_index")
    if isinstance(task_stats, dict):
        task_stats["min"] = [int(task_index)]
        task_stats["max"] = [int(task_index)]
        task_stats["mean"] = [float(task_index)]
        task_stats["std"] = [0.0]
        task_stats["count"] = [int(length)]
    return row


def _rewrite_metadata_and_parquet(
    *,
    dataset_dir: Path,
    episode_rows: list[dict[str, Any]],
    prompts_by_episode: dict[int, str],
    dry_run: bool,
) -> dict[str, Any]:
    meta_dir = dataset_dir / "meta"
    info_path = meta_dir / "info.json"
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"

    info = _read_json(info_path)
    if info["codebase_version"] == "v3.0" and not dry_run:
        from data_analysis.dataset_io import rewrite_v3_tasks

        return rewrite_v3_tasks(dataset_dir, prompts_by_episode)

    task_to_index: dict[str, int] = {}
    tasks_rows: list[dict[str, Any]] = []
    episode_task_indices: dict[int, int] = {}

    new_episode_rows: list[dict[str, Any]] = []
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        prompt = prompts_by_episode[episode_index]
        if prompt not in task_to_index:
            task_to_index[prompt] = len(task_to_index)
            tasks_rows.append(
                {
                    "task_index": task_to_index[prompt],
                    "task": prompt,
                }
            )
        task_index = task_to_index[prompt]
        episode_task_indices[episode_index] = task_index

        new_row = dict(row)
        new_row["tasks"] = [prompt]
        new_episode_rows.append(new_row)

    if dry_run:
        return {
            "num_tasks": len(tasks_rows),
            "episode_task_indices": episode_task_indices,
            "tasks": tasks_rows,
        }

    _write_jsonl(tasks_path, tasks_rows)
    _write_jsonl(episodes_path, new_episode_rows)

    for row in tqdm(new_episode_rows, desc="Rewriting parquet task_index"):
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        parquet_path = _episode_parquet_path(info, dataset_dir, episode_index)
        _rewrite_parquet_task_index(
            path=parquet_path,
            task_index=episode_task_indices[episode_index],
            expected_length=length,
        )

    if stats_path.exists():
        stats_rows = _read_jsonl(stats_path)
        length_by_episode = {
            int(row["episode_index"]): int(row["length"]) for row in new_episode_rows
        }
        patched_stats = []
        for row in stats_rows:
            episode_index = int(row.get("episode_index", -1))
            patched_stats.append(
                _patch_task_index_stats(
                    row,
                    task_index=episode_task_indices[episode_index],
                    length=length_by_episode[episode_index],
                )
            )
        _write_jsonl(stats_path, patched_stats)

    info["total_tasks"] = len(tasks_rows)
    _write_json(info_path, info)

    return {
        "num_tasks": len(tasks_rows),
        "episode_task_indices": episode_task_indices,
        "tasks": tasks_rows,
    }


def convert_dataset(args: argparse.Namespace) -> int:
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

    print(f"Input dataset:  {input_dir}")
    print(f"Output dataset: {output_dir}")
    print(f"ASR model:      {args.asr_model}")
    print(f"ASR device:     {args.asr_device}")
    print(f"ASR language:   {args.asr_language or 'auto'}")

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

    for row in tqdm(episode_rows, desc="Transcribing VAD clips"):
        episode_index = int(row["episode_index"])
        prompt, metadata = _collect_episode_prompt(
            dataset_dir=work_dir,
            episode_row=row,
            model_name=args.asr_model,
            device=args.asr_device,
            language=args.asr_language,
            max_new_tokens=args.asr_max_new_tokens,
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
    print("ASR conversion summary")
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
        description="Copy a LeRobot audio dataset and replace prompts with Whisper ASR text."
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
        help="Output directory. Defaults to <input_dir>_asr.",
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
        "--asr-model",
        type=str,
        default="openai/whisper-base",
        help="Whisper ASR model name or local model path.",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="auto",
        help="Whisper ASR device, e.g. auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--asr-language",
        type=str,
        default="chinese",
        help="Optional Whisper language, e.g. english or chinese.",
    )
    parser.add_argument(
        "--asr-max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens generated by Whisper.",
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
