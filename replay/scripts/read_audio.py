"""Audio metadata, waveform and offset on the shared episode timeline."""

import wave

import numpy as np

from ._common import _get_episode, _integer, _load_metadata
from .read_sidecars import confined, origin_ns, read_sync


def resolve_audio(root, episode_index):
    _, episodes = _load_metadata(root)
    _get_episode(episodes, episode_index)
    sync = read_sync(root, episode_index)
    path = confined(root, sync.get("audio_path", f"audio/episode_{episode_index:06d}.wav"))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, sync


def read_audio(root, episode_index, max_points=1000):
    _integer(max_points, "max_points", minimum=2)
    path, sync = resolve_audio(root, episode_index)
    with wave.open(str(path), "rb") as wav:
        channels, rate, width, count = (
            wav.getnchannels(),
            wav.getframerate(),
            wav.getsampwidth(),
            wav.getnframes(),
        )
        if width not in (1, 2, 4):
            raise ValueError("Unsupported WAV PCM width")
        samples = np.frombuffer(
            wav.readframes(count), dtype={1: np.uint8, 2: np.int16, 4: np.int32}[width]
        ).astype(float)
    if width == 1:
        samples -= 128
    samples = samples.reshape(-1, channels).mean(axis=1) / (2 ** (width * 8 - 1))
    start = (int(sync.get("audio_start_monotonic_ns", origin_ns(sync))) - origin_ns(sync)) / 1e9
    chunks = np.array_split(samples, min(max_points, max(1, len(samples))))
    waveform = [float(np.max(np.abs(chunk))) if len(chunk) else 0.0 for chunk in chunks]
    return {
        "sample_rate": rate,
        "channels": channels,
        "num_samples": count,
        "duration_s": count / rate,
        "offset_s": start,
        "waveform": waveform,
        "vad_segments": sync.get("vad_segments", []),
        "instruction_audio_window": sync.get("instruction_audio_window"),
        "label": sync.get("label"),
        "asr_metadata": sync.get("asr_metadata"),
    }
