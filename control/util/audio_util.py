from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import wave

import numpy as np


@dataclass(frozen=True)
class EpisodeAudioSegment:
    audio: np.ndarray
    sample_rate: int
    audio_path: Path
    sync_path: Path | None
    start_sample: int
    end_sample: int
    duration_s: float


def _episode_audio_stem(episode_index: int) -> str:
    return f"episode_{int(episode_index):06d}"


def _resolve_episode_audio_paths(
    ds_root: Path,
    episode_index: int,
) -> tuple[Path, Path]:
    audio_dir = ds_root / "audio"
    stem = _episode_audio_stem(episode_index)
    return audio_dir / f"{stem}.wav", audio_dir / f"{stem}.sync.json"


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Replay] Warning: failed to read JSON {path}: {exc}")
        return None


def _resolve_audio_path_from_sync(ds_root: Path, sync_data: dict) -> Path | None:
    raw_audio_path = sync_data.get("audio_path")
    if not raw_audio_path:
        return None

    audio_path = Path(str(raw_audio_path))
    if audio_path.is_absolute():
        return audio_path
    return ds_root / audio_path


def read_wav_pcm(
    path: Path,
    require_nonempty: bool = False,
) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = int(wav_file.getnchannels())
        sample_rate = int(wav_file.getframerate())
        sample_width = int(wav_file.getsampwidth())
        num_frames = int(wav_file.getnframes())
        raw = wav_file.readframes(num_frames)

    dtype_map = {
        1: np.uint8,
        2: np.int16,
        4: np.int32,
    }
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes ({path})")

    audio = np.frombuffer(raw, dtype=dtype_map[sample_width])
    if channels > 1:
        audio = audio.reshape(-1, channels)
    else:
        audio = audio.reshape(-1, 1)

    if require_nonempty and audio.size == 0:
        raise RuntimeError(f"Recorded WAV is empty: {path}")

    if audio.dtype == np.uint8:
        audio = ((audio.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    else:
        audio = np.ascontiguousarray(audio)
    return audio, sample_rate


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    return read_wav_pcm(path)


def play_audio(
    audio: np.ndarray,
    *,
    sample_rate: int,
    blocking: bool = True,
) -> float:
    import sounddevice as sd

    playback_audio = np.ascontiguousarray(audio)
    if playback_audio.ndim == 1:
        playback_audio = playback_audio.reshape(-1, 1)
    if playback_audio.size == 0:
        raise RuntimeError("Cannot play empty audio")

    dtype_name = np.dtype(playback_audio.dtype).name
    output_channels = int(playback_audio.shape[1])

    try:
        sd.check_output_settings(
            samplerate=sample_rate,
            channels=output_channels,
            dtype=dtype_name,
        )
    except Exception:
        if output_channels != 1:
            raise
        playback_audio = np.repeat(playback_audio, 2, axis=1)
        output_channels = 2
        sd.check_output_settings(
            samplerate=sample_rate,
            channels=output_channels,
            dtype=np.dtype(playback_audio.dtype).name,
        )

    sd.play(playback_audio, samplerate=sample_rate, blocking=blocking)
    return float(audio.shape[0]) / float(sample_rate)


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _get_frame_record(frame_records: list[dict], frame_index: int) -> dict | None:
    if 0 <= frame_index < len(frame_records):
        record = frame_records[frame_index]
        if _safe_int(record.get("frame_index")) in (None, frame_index):
            return record

    for record in frame_records:
        if _safe_int(record.get("frame_index")) == frame_index:
            return record
    return None


def _frame_range_to_audio_samples(
    *,
    sync_data: dict | None,
    start_frame_index: int,
    end_frame_index: int,
    sample_rate: int,
    audio_total_samples: int,
    fps: float,
) -> tuple[int, int]:
    fps_safe = max(float(fps), 1e-6)
    fallback_start = int(
        np.clip(
            round((start_frame_index / fps_safe) * sample_rate), 0, audio_total_samples
        )
    )
    fallback_end = int(
        np.clip(
            round((end_frame_index / fps_safe) * sample_rate), 0, audio_total_samples
        )
    )

    if not sync_data:
        return fallback_start, max(fallback_start, fallback_end)

    audio_start_ns = _safe_int(sync_data.get("audio_start_monotonic_ns"))
    frame_records = sync_data.get("frame_records")
    if (
        audio_start_ns is None
        or not isinstance(frame_records, list)
        or not frame_records
    ):
        return fallback_start, max(fallback_start, fallback_end)

    start_record = _get_frame_record(frame_records, start_frame_index)
    end_record = _get_frame_record(frame_records, end_frame_index - 1)
    next_record = _get_frame_record(frame_records, end_frame_index)
    if start_record is None or end_record is None:
        return fallback_start, max(fallback_start, fallback_end)

    start_ns = _safe_int(start_record.get("host_frame_monotonic_ns"))
    end_ns = _safe_int(next_record.get("host_frame_monotonic_ns"))
    if end_ns is None:
        last_ns = _safe_int(end_record.get("host_frame_monotonic_ns"))
        if last_ns is not None:
            end_ns = last_ns + int(round(1e9 / fps_safe))

    if start_ns is None or end_ns is None or end_ns <= start_ns:
        return fallback_start, max(fallback_start, fallback_end)

    start_sample = int(
        np.clip(
            round(((start_ns - audio_start_ns) / 1e9) * sample_rate),
            0,
            audio_total_samples,
        )
    )
    end_sample = int(
        np.clip(
            round(((end_ns - audio_start_ns) / 1e9) * sample_rate),
            0,
            audio_total_samples,
        )
    )
    return start_sample, max(start_sample, end_sample)


def load_episode_audio_segment(
    *,
    ds_root: Path,
    episode_index: int,
    start_frame_index: int,
    end_frame_index: int,
    fps: float,
) -> EpisodeAudioSegment | None:
    default_audio_path, sync_path = _resolve_episode_audio_paths(ds_root, episode_index)
    sync_data = _load_json(sync_path)
    audio_path = default_audio_path
    if sync_data is not None:
        sync_audio_path = _resolve_audio_path_from_sync(ds_root, sync_data)
        if sync_audio_path is not None:
            audio_path = sync_audio_path

    if not audio_path.is_file():
        print(
            f"[Replay] Warning: audio file not found for episode {episode_index}: {audio_path}"
        )
        return None

    try:
        audio, sample_rate = read_wav_pcm(audio_path)
    except Exception as exc:
        print(f"[Replay] Warning: failed to read audio {audio_path}: {exc}")
        return None

    start_sample, end_sample = _frame_range_to_audio_samples(
        sync_data=sync_data,
        start_frame_index=start_frame_index,
        end_frame_index=end_frame_index,
        sample_rate=sample_rate,
        audio_total_samples=int(audio.shape[0]),
        fps=fps,
    )
    segment = np.ascontiguousarray(audio[start_sample:end_sample])
    if segment.size == 0:
        print(
            f"[Replay] Warning: audio slice is empty for episode {episode_index} "
            f"(samples {start_sample}:{end_sample})"
        )
        return None

    return EpisodeAudioSegment(
        audio=segment,
        sample_rate=sample_rate,
        audio_path=audio_path,
        sync_path=sync_path if sync_path.is_file() else None,
        start_sample=start_sample,
        end_sample=end_sample,
        duration_s=float(segment.shape[0]) / float(sample_rate),
    )


def start_audio_playback(
    audio_segment: EpisodeAudioSegment | None,
    *,
    speed: float = 1.0,
) -> bool:
    if audio_segment is None:
        return False

    try:
        import sounddevice as sd
    except Exception as exc:
        print(
            f"[Replay] Warning: sounddevice unavailable, skipping audio playback: {exc}"
        )
        return False

    playback_rate = max(1, int(round(float(audio_segment.sample_rate) * float(speed))))
    try:
        sd.stop()
        play_audio(audio_segment.audio, sample_rate=playback_rate, blocking=False)
    except Exception as exc:
        print(f"[Replay] Warning: failed to start audio playback: {exc}")
        return False
    return True


def stop_audio_playback() -> None:
    try:
        import sounddevice as sd
    except Exception:
        return
    try:
        sd.stop()
    except Exception:
        pass
