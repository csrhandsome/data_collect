from __future__ import annotations

import functools
import json
import math
import shutil
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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


def default_microphone_output_path(*, base_dir: Path | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path.cwd() if base_dir is None else Path(base_dir)
    return root / "recordings" / f"microphone_{timestamp}.wav"


def block_frames_for_sample_rate(sample_rate: int, *, block_duration_s: float = 0.05) -> int:
    return max(256, int(round(int(sample_rate) * float(block_duration_s))))


def metadata_path_for_audio_output(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_suffix(".audio.json")
    return output_path.with_name(f"{output_path.name}.audio.json")


def empty_audio_normalization_info(
    *,
    target_peak_ratio: float,
    reason: str,
) -> dict:
    return {
        "applied": False,
        "reason": reason,
        "gain": None,
        "original_peak": None,
        "target_peak": None,
        "target_peak_ratio": float(target_peak_ratio),
    }


def build_audio_recording_metadata(
    *,
    output_path: Path,
    sample_rate: int,
    channels: int,
    dtype: np.dtype,
    input_device: int | str | None,
    normalize_audio: bool,
    normalization_target_peak_ratio: float,
    audio_start_monotonic_ns: int | None,
    audio_stop_monotonic_ns: int | None,
    chunk_timestamps_ns: list[int],
    input_buffer_adc_times: list[float | None],
    input_overflow_timestamps_ns: list[int],
    num_samples: int,
    normalization_info: dict | None = None,
) -> dict:
    info = normalization_info or empty_audio_normalization_info(
        target_peak_ratio=normalization_target_peak_ratio,
        reason="unknown",
    )
    return {
        "audio_path": str(output_path),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "dtype": np.dtype(dtype).name,
        "input_device": input_device,
        "normalize_audio": bool(normalize_audio),
        "normalization_applied": info.get("applied"),
        "normalization_reason": info.get("reason"),
        "normalization_gain": info.get("gain"),
        "normalization_original_peak": info.get("original_peak"),
        "normalization_target_peak": info.get("target_peak"),
        "normalization_target_peak_ratio": info.get("target_peak_ratio"),
        "audio_start_monotonic_ns": audio_start_monotonic_ns,
        "audio_stop_monotonic_ns": audio_stop_monotonic_ns,
        "num_chunks": len(chunk_timestamps_ns),
        "num_samples": int(num_samples),
        "read_block_frames": block_frames_for_sample_rate(sample_rate),
        "num_input_overflows": len(input_overflow_timestamps_ns),
        "chunk_timestamps_ns": list(chunk_timestamps_ns),
        "input_buffer_adc_times": list(input_buffer_adc_times),
        "input_overflow_timestamps_ns": list(input_overflow_timestamps_ns),
    }


def write_audio_recording_metadata(
    *,
    output_path: Path,
    sample_rate: int,
    channels: int,
    dtype: np.dtype,
    input_device: int | str | None,
    normalize_audio: bool,
    normalization_target_peak_ratio: float,
    audio_start_monotonic_ns: int | None,
    audio_stop_monotonic_ns: int | None,
    chunk_timestamps_ns: list[int],
    input_buffer_adc_times: list[float | None],
    input_overflow_timestamps_ns: list[int],
    num_samples: int,
    normalization_info: dict | None = None,
) -> tuple[Path, dict]:
    metadata_path = metadata_path_for_audio_output(output_path)
    payload = build_audio_recording_metadata(
        output_path=output_path,
        sample_rate=sample_rate,
        channels=channels,
        dtype=dtype,
        input_device=input_device,
        normalize_audio=normalize_audio,
        normalization_target_peak_ratio=normalization_target_peak_ratio,
        audio_start_monotonic_ns=audio_start_monotonic_ns,
        audio_stop_monotonic_ns=audio_stop_monotonic_ns,
        chunk_timestamps_ns=chunk_timestamps_ns,
        input_buffer_adc_times=input_buffer_adc_times,
        input_overflow_timestamps_ns=input_overflow_timestamps_ns,
        num_samples=num_samples,
        normalization_info=normalization_info,
    )
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata_path, payload


def write_normalized_wav_from_existing(
    *,
    source_path: Path,
    output_path: Path,
    sample_rate: int,
    channels: int,
    dtype: np.dtype,
    normalization_target_peak_ratio: float,
) -> dict:
    audio, _ = read_wav_pcm(source_path)
    pcm = np.ascontiguousarray(audio.astype(dtype, copy=False))
    normalized_pcm, normalization_info = normalize_pcm_audio(
        pcm,
        target_peak_ratio=normalization_target_peak_ratio,
    )
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(int(channels))
        wav_file.setsampwidth(np.dtype(dtype).itemsize)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(normalized_pcm.tobytes())
    return normalization_info


def discard_staged_audio_file(
    staged_wav_path: Path | None,
    *,
    staged_is_temp: bool,
) -> None:
    if staged_wav_path is None or not staged_is_temp:
        return
    try:
        staged_wav_path.unlink(missing_ok=True)
    except Exception:
        pass


def save_staged_audio_recording(
    *,
    staged_wav_path: Path,
    staged_is_temp: bool,
    output_path: Path,
    sample_rate: int,
    channels: int,
    dtype: np.dtype,
    input_device: int | str | None,
    normalize_audio: bool,
    normalization_target_peak_ratio: float,
    audio_start_monotonic_ns: int | None,
    audio_stop_monotonic_ns: int | None,
    chunk_timestamps_ns: list[int],
    input_buffer_adc_times: list[float | None],
    input_overflow_timestamps_ns: list[int],
    num_samples: int,
) -> tuple[Path, Path, dict]:
    if not staged_wav_path.is_file():
        raise FileNotFoundError(f"Staged WAV not found: {staged_wav_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize_audio:
        normalization_info = write_normalized_wav_from_existing(
            source_path=staged_wav_path,
            output_path=output_path,
            sample_rate=sample_rate,
            channels=channels,
            dtype=dtype,
            normalization_target_peak_ratio=normalization_target_peak_ratio,
        )
        discard_staged_audio_file(staged_wav_path, staged_is_temp=staged_is_temp)
    else:
        if staged_is_temp:
            shutil.move(str(staged_wav_path), str(output_path))
        elif staged_wav_path != output_path:
            shutil.copyfile(str(staged_wav_path), str(output_path))
        normalization_info = empty_audio_normalization_info(
            target_peak_ratio=normalization_target_peak_ratio,
            reason="disabled",
        )

    metadata_path, metadata = write_audio_recording_metadata(
        output_path=output_path,
        sample_rate=sample_rate,
        channels=channels,
        dtype=dtype,
        input_device=input_device,
        normalize_audio=normalize_audio,
        normalization_target_peak_ratio=normalization_target_peak_ratio,
        audio_start_monotonic_ns=audio_start_monotonic_ns,
        audio_stop_monotonic_ns=audio_stop_monotonic_ns,
        chunk_timestamps_ns=chunk_timestamps_ns,
        input_buffer_adc_times=input_buffer_adc_times,
        input_overflow_timestamps_ns=input_overflow_timestamps_ns,
        num_samples=num_samples,
        normalization_info=normalization_info,
    )
    return output_path, metadata_path, metadata


def _looks_like_usb_input_device(device_name: object) -> bool:
    name = str(device_name).strip().lower()
    if not name:
        return False
    usb_markers = ("usb", "mic", "microphone", "headset", "composite")
    return any(marker in name for marker in usb_markers)


def _resolve_preferred_input_device() -> int | str | None:
    import sounddevice as sd

    try:
        default_input, _ = sd.default.device
    except Exception:
        default_input = None

    try:
        devices = sd.query_devices()
    except Exception:
        return default_input if default_input not in (-1, None) else None

    valid_default = default_input if default_input not in (-1, None) else None
    if isinstance(valid_default, int):
        try:
            default_info = devices[valid_default]
            if int(default_info.get("max_input_channels", 0)) > 0 and _looks_like_usb_input_device(
                default_info.get("name")
            ):
                return valid_default
        except Exception:
            pass

    for index, device_info in enumerate(devices):
        try:
            if int(device_info.get("max_input_channels", 0)) <= 0:
                continue
        except Exception:
            continue
        if _looks_like_usb_input_device(device_info.get("name")):
            return index

    return valid_default


def _resolve_supported_input_sample_rate(
    *,
    input_device: int | str | None,
    requested_sample_rate: int,
    channels: int,
    dtype_name: str,
) -> int:
    import sounddevice as sd

    requested_rate = int(requested_sample_rate)
    try:
        sd.check_input_settings(
            device=input_device,
            samplerate=requested_rate,
            channels=channels,
            dtype=dtype_name,
        )
        return requested_rate
    except Exception:
        pass

    try:
        device_info = sd.query_devices(input_device)
        fallback_rate = int(round(float(device_info["default_samplerate"])))
    except Exception:
        sd.check_input_settings(
            device=input_device,
            samplerate=requested_rate,
            channels=channels,
            dtype=dtype_name,
        )
        return requested_rate

    if fallback_rate <= 0 or fallback_rate == requested_rate:
        sd.check_input_settings(
            device=input_device,
            samplerate=requested_rate,
            channels=channels,
            dtype=dtype_name,
        )
        return requested_rate

    sd.check_input_settings(
        device=input_device,
        samplerate=fallback_rate,
        channels=channels,
        dtype=dtype_name,
    )
    print(
        "[MicrophoneRecorder] "
        f"Requested sample rate {requested_rate} Hz unsupported on input device "
        f"{input_device}; using {fallback_rate} Hz instead."
    )
    return fallback_rate


def normalize_pcm_audio(
    audio: np.ndarray,
    *,
    target_peak_ratio: float = 0.95,
) -> tuple[np.ndarray, dict]:
    pcm = np.ascontiguousarray(audio)
    if pcm.dtype.kind not in {"i", "u"}:
        raise ValueError("Peak normalization only supports integer PCM audio")

    ratio = float(target_peak_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError("target_peak_ratio must be in (0, 1]")

    if pcm.size == 0:
        return pcm, {
            "applied": False,
            "reason": "empty",
            "gain": None,
            "original_peak": 0.0,
            "target_peak": None,
            "target_peak_ratio": ratio,
        }

    dtype_info = np.iinfo(pcm.dtype)
    pcm_float = pcm.astype(np.float32, copy=False)
    if pcm.dtype.kind == "u":
        midpoint = float(dtype_info.max + 1) / 2.0
        centered = pcm_float - midpoint
        original_peak = float(np.max(np.abs(centered)))
        target_peak = midpoint * ratio
        if original_peak <= 0.0:
            return pcm, {
                "applied": False,
                "reason": "silent",
                "gain": None,
                "original_peak": original_peak,
                "target_peak": target_peak,
                "target_peak_ratio": ratio,
            }
        gain = target_peak / original_peak
        normalized = np.clip(
            np.rint(centered * gain + midpoint), dtype_info.min, dtype_info.max
        ).astype(pcm.dtype)
    else:
        original_peak = float(np.max(np.abs(pcm_float)))
        target_peak = float(dtype_info.max) * ratio
        if original_peak <= 0.0:
            return pcm, {
                "applied": False,
                "reason": "silent",
                "gain": None,
                "original_peak": original_peak,
                "target_peak": target_peak,
                "target_peak_ratio": ratio,
            }
        gain = target_peak / original_peak
        normalized = np.clip(
            np.rint(pcm_float * gain), dtype_info.min, dtype_info.max
        ).astype(pcm.dtype)

    return np.ascontiguousarray(normalized), {
        "applied": abs(gain - 1.0) > 1e-6,
        "reason": "ok",
        "gain": float(gain),
        "original_peak": original_peak,
        "target_peak": float(target_peak),
        "target_peak_ratio": ratio,
    }


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


def _resolve_whisper_device(device: str) -> str:
    if device != "auto":
        return device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=8)
def _load_whisper_asr_components(model_name: str, device: str):
    import torch

    from control.util.hub_compat import allow_hub1_for_transformers

    allow_hub1_for_transformers()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    resolved_device = _resolve_whisper_device(device)
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    model.to(resolved_device)
    if resolved_device.startswith("cuda"):
        model.to(dtype=torch.float16)
    return processor, model, resolved_device


def _audio_to_float32_mono(audio: np.ndarray) -> np.ndarray:
    audio_arr = np.asarray(audio)
    if audio_arr.ndim == 2:
        audio_arr = audio_arr.mean(axis=1)
    else:
        audio_arr = audio_arr.reshape(-1)

    if audio_arr.dtype == np.float32:
        return np.ascontiguousarray(audio_arr)
    if audio_arr.dtype.kind == "f":
        return np.ascontiguousarray(audio_arr.astype(np.float32))
    if audio_arr.dtype.kind in {"i", "u"}:
        if audio_arr.dtype == np.uint8:
            normalized = (audio_arr.astype(np.float32) - 128.0) / 128.0
            return np.ascontiguousarray(normalized.astype(np.float32))
        dtype_info = np.iinfo(audio_arr.dtype)
        scale = float(max(abs(dtype_info.min), dtype_info.max))
        normalized = audio_arr.astype(np.float32) / scale
        return np.ascontiguousarray(normalized.astype(np.float32))
    raise ValueError(f"Unsupported audio dtype for ASR: {audio_arr.dtype}")


def _resample_audio(
    audio: np.ndarray,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    source_rate = int(source_sample_rate)
    target_rate = int(target_sample_rate)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be > 0")
    if source_rate == target_rate or audio.size == 0:
        return np.ascontiguousarray(audio, dtype=np.float32)

    from scipy.signal import resample_poly

    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    resampled = resample_poly(audio, up, down).astype(np.float32, copy=False)
    return np.ascontiguousarray(resampled)


def transcribe_whisper_asr(
    audio: np.ndarray,
    *,
    sample_rate: int,
    model_name: str = "openai/whisper-base",
    device: str = "auto",
    language: str | None = None,
    task: str = "transcribe",
    max_new_tokens: int = 128,
) -> str:
    """Transcribe a waveform with Whisper ASR and return decoded text."""
    waveform = _audio_to_float32_mono(audio)
    if waveform.size == 0:
        return ""

    processor, model, resolved_device = _load_whisper_asr_components(model_name, device)
    target_sample_rate = int(processor.feature_extractor.sampling_rate)
    waveform = _resample_audio(
        waveform,
        source_sample_rate=int(sample_rate),
        target_sample_rate=target_sample_rate,
    )

    import torch

    inputs = processor(waveform, sampling_rate=target_sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(resolved_device)
    if resolved_device.startswith("cuda"):
        input_features = input_features.to(dtype=torch.float16)

    generate_kwargs: dict[str, object] = {"max_new_tokens": int(max_new_tokens)}
    if language:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

    with torch.inference_mode():
        predicted_ids = model.generate(input_features, **generate_kwargs)

    texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return texts[0].strip() if texts else ""


def transcribe_whisper_asr_from_wav(
    path: Path,
    *,
    model_name: str = "openai/whisper-base",
    device: str = "auto",
    language: str | None = None,
    task: str = "transcribe",
    max_new_tokens: int = 128,
) -> str:
    audio, sample_rate = read_wav_pcm(path, require_nonempty=True)
    return transcribe_whisper_asr(
        audio,
        sample_rate=sample_rate,
        model_name=model_name,
        device=device,
        language=language,
        task=task,
        max_new_tokens=max_new_tokens,
    )


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
    end_ns = (
        _safe_int(next_record.get("host_frame_monotonic_ns"))
        if next_record is not None
        else None
    )
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
