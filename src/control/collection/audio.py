"""Audio episode metadata and an explicitly synthetic recorder for dry runs."""

import json
import time
import wave
from pathlib import Path

import numpy as np


class SyntheticMicrophone:
    def __init__(self, config):
        self.sample_rate = int(config.get("sample_rate", 16000))
        self.channels = int(config.get("channels", 1))
        self.default_output_path = self.last_metadata_path = None
        self.audio_start_monotonic_ns = self.audio_stop_monotonic_ns = None
        self.is_recording = False

    def start_recording(self):
        self.audio_start_monotonic_ns = time.monotonic_ns()
        self.is_recording = True
        return True

    def stop_recording(self):
        if not self.is_recording:
            return None
        self.audio_stop_monotonic_ns = time.monotonic_ns()
        self.is_recording = False
        if self.default_output_path is None:
            return None
        path = Path(self.default_output_path)
        frames = max(
            1,
            round(
                (self.audio_stop_monotonic_ns - self.audio_start_monotonic_ns)
                / 1e9
                * self.sample_rate
            ),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(np.zeros((frames, self.channels), dtype=np.int16).tobytes())
        self.last_metadata_path = path.with_suffix(".audio.json")
        self.last_metadata_path.write_text(
            json.dumps(
                {
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "num_samples": frames,
                    "synthetic": True,
                    "audio_start_monotonic_ns": self.audio_start_monotonic_ns,
                    "audio_stop_monotonic_ns": self.audio_stop_monotonic_ns,
                }
            )
        )
        return path

    def stop(self):
        self.default_output_path = None
        self.stop_recording()
