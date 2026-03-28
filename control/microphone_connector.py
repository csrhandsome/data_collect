from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import json
import threading
import time
from tempfile import TemporaryDirectory
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from .util.audio_util import play_audio, read_wav_pcm


class MicrophoneRecorder:
    """Pure microphone recorder with episode-friendly WAV + JSON outputs."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: np.dtype | type[np.generic] = np.int16,
        default_output_path: Optional[Path] = None,
    ) -> None:
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._dtype = np.dtype(dtype)
        if self._channels <= 0:
            raise ValueError("channels must be > 0")
        if self._sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self._dtype.kind not in {"i", "u"}:
            raise ValueError("Only integer PCM dtypes are supported for WAV output")

        self._default_output_path = (
            Path(default_output_path) if default_output_path is not None else None
        )
        self._frames: list[np.ndarray] = []
        self._chunk_timestamps_ns: list[int] = []
        self._input_buffer_adc_times: list[Optional[float]] = []
        self._buffer_lock = threading.Lock()
        self._is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self._audio_start_monotonic_ns: Optional[int] = None
        self._audio_stop_monotonic_ns: Optional[int] = None
        self._last_saved_path: Optional[Path] = None
        self._last_metadata_path: Optional[Path] = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def frames(self) -> tuple[np.ndarray, ...]:
        with self._buffer_lock:
            return tuple(self._frames)

    @property
    def chunk_timestamps_ns(self) -> tuple[int, ...]:
        with self._buffer_lock:
            return tuple(self._chunk_timestamps_ns)

    @property
    def input_buffer_adc_times(self) -> tuple[Optional[float], ...]:
        with self._buffer_lock:
            return tuple(self._input_buffer_adc_times)

    @property
    def audio_start_monotonic_ns(self) -> Optional[int]:
        return self._audio_start_monotonic_ns

    @property
    def audio_stop_monotonic_ns(self) -> Optional[int]:
        return self._audio_stop_monotonic_ns

    @property
    def default_output_path(self) -> Optional[Path]:
        return self._default_output_path

    @default_output_path.setter
    def default_output_path(self, value: Optional[Path]) -> None:
        self._default_output_path = Path(value) if value is not None else None

    @property
    def last_saved_path(self) -> Optional[Path]:
        return self._last_saved_path

    @property
    def last_metadata_path(self) -> Optional[Path]:
        return self._last_metadata_path

    def reset(self) -> None:
        if self._is_recording or (
            self.recording_thread is not None and self.recording_thread.is_alive()
        ):
            raise RuntimeError("Cannot reset while recording is active")

        with self._buffer_lock:
            self._frames.clear()
            self._chunk_timestamps_ns.clear()
            self._input_buffer_adc_times.clear()
        self._audio_start_monotonic_ns = None
        self._audio_stop_monotonic_ns = None
        self._last_saved_path = None
        self._last_metadata_path = None

    def start_recording(self) -> bool:
        if self._is_recording:
            return False

        try:
            self.reset()
        except RuntimeError:
            return False

        self._is_recording = True
        self._audio_start_monotonic_ns = time.monotonic_ns()

        try:
            self.recording_thread = threading.Thread(
                target=self._record_audio,
                daemon=True,
            )
            self.recording_thread.start()
            return True
        except Exception as exc:
            print(f"[MicrophoneRecorder] Failed to start recording thread: {exc}")
            self._is_recording = False
            self.recording_thread = None
            return False

    def stop_recording(self) -> Optional[Path]:
        if not self._is_recording and self.recording_thread is None:
            return None

        self._is_recording = False
        if self.recording_thread is not None:
            self.recording_thread.join()
            self.recording_thread = None

        self._audio_stop_monotonic_ns = time.monotonic_ns()

        if self._default_output_path is None:
            return None
        return self.save_recording(self._default_output_path)

    def save_recording(self, output_path: Path) -> Path:
        if self._is_recording:
            raise RuntimeError("Cannot save recording while recording is active")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pcm = self._concat_frames()
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(self._channels)
            wav_file.setsampwidth(self._dtype.itemsize)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(pcm.tobytes())

        metadata_path = self._metadata_path_for(output_path)
        metadata = self._build_metadata(output_path=output_path, num_samples=int(pcm.shape[0]))
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._last_saved_path = output_path
        self._last_metadata_path = metadata_path
        return output_path

    def _record_audio(self) -> None:
        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype=self._dtype.name,
                callback=self._audio_callback,
            ):
                while self._is_recording:
                    sd.sleep(100)
        except Exception as exc:
            print(f"[MicrophoneRecorder] Recording error: {exc}")
            self._is_recording = False

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        del frames, status
        if not self._is_recording:
            return

        chunk = np.asarray(indata.copy(), dtype=self._dtype)
        chunk_timestamp_ns = time.monotonic_ns()

        input_buffer_adc_time: Optional[float] = None
        if time_info is not None:
            adc_time = getattr(time_info, "inputBufferAdcTime", None)
            if adc_time is not None:
                input_buffer_adc_time = float(adc_time)

        with self._buffer_lock:
            self._frames.append(chunk)
            self._chunk_timestamps_ns.append(chunk_timestamp_ns)
            self._input_buffer_adc_times.append(input_buffer_adc_time)

    def _concat_frames(self) -> np.ndarray:
        with self._buffer_lock:
            if not self._frames:
                return np.empty((0, self._channels), dtype=self._dtype)
            frames = [np.asarray(frame, dtype=self._dtype) for frame in self._frames]
        return np.ascontiguousarray(np.concatenate(frames, axis=0))

    def _metadata_path_for(self, output_path: Path) -> Path:
        if output_path.suffix:
            return output_path.with_suffix(".audio.json")
        return output_path.with_name(f"{output_path.name}.audio.json")

    def _build_metadata(self, *, output_path: Path, num_samples: int) -> dict:
        with self._buffer_lock:
            chunk_timestamps_ns = list(self._chunk_timestamps_ns)
            input_buffer_adc_times = list(self._input_buffer_adc_times)

        return {
            "audio_path": str(output_path),
            "sample_rate": self._sample_rate,
            "channels": self._channels,
            "dtype": self._dtype.name,
            "audio_start_monotonic_ns": self._audio_start_monotonic_ns,
            "audio_stop_monotonic_ns": self._audio_stop_monotonic_ns,
            "num_chunks": len(chunk_timestamps_ns),
            "num_samples": num_samples,
            "chunk_timestamps_ns": chunk_timestamps_ns,
            "input_buffer_adc_times": input_buffer_adc_times,
        }


def _default_saved_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "recordings" / f"microphone_{timestamp}.wav"


def run_microphone_recorder_smoke_test(
    *,
    record_seconds: float = 5.0,
    playback_seconds: float = 5.0,
    sample_rate: int = 16000,
    channels: int = 1,
    save: bool = False,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Hardware smoke test: record from the mic, optionally save, then play back."""

    if record_seconds <= 0:
        raise ValueError("record_seconds must be > 0")
    if playback_seconds < 0:
        raise ValueError("playback_seconds must be >= 0")

    persistent_output_path = (
        Path(output_path)
        if output_path is not None
        else (_default_saved_output_path() if save else None)
    )

    temp_dir_ctx: TemporaryDirectory[str] | contextlib.AbstractContextManager[None]
    if persistent_output_path is None:
        temp_dir_ctx = TemporaryDirectory(prefix="microphone_recorder_test_")
    else:
        temp_dir_ctx = contextlib.nullcontext()

    with temp_dir_ctx as tmp_dir:
        resolved_output_path = (
            persistent_output_path
            if persistent_output_path is not None
            else Path(tmp_dir) / "microphone_test.wav"
        )
        recorder = MicrophoneRecorder(
            sample_rate=sample_rate,
            channels=channels,
            default_output_path=resolved_output_path,
        )
        saved_path: Optional[Path] = None
        metadata_path: Optional[Path] = None

        try:
            sd.check_input_settings(
                samplerate=recorder.sample_rate,
                channels=recorder.channels,
                dtype=recorder.dtype.name,
            )

            save_mode = "persistent save" if persistent_output_path is not None else "temp"
            print(
                f"[MicrophoneRecorder Test] Recording for {record_seconds:.1f}s "
                f"to {resolved_output_path} ({save_mode})"
            )
            if not recorder.start_recording():
                raise RuntimeError("Failed to start microphone recording")

            time.sleep(record_seconds)
            saved_path = recorder.stop_recording()
            metadata_path = recorder.last_metadata_path
            if saved_path is None or not saved_path.is_file():
                raise RuntimeError("Recording did not produce a WAV file")

            if playback_seconds > 0:
                audio, saved_sample_rate = read_wav_pcm(saved_path, require_nonempty=True)
                recorded_duration_s = float(audio.shape[0]) / float(saved_sample_rate)
                max_playback_samples = max(
                    1, int(round(playback_seconds * saved_sample_rate))
                )
                playback_audio = np.ascontiguousarray(audio[:max_playback_samples])

                print(
                    f"[MicrophoneRecorder Test] Recorded {recorded_duration_s:.2f}s, "
                    f"playing back {min(recorded_duration_s, playback_seconds):.2f}s"
                )
                play_audio(playback_audio, sample_rate=saved_sample_rate, blocking=True)
        finally:
            if recorder.is_recording:
                recorder.stop_recording()
            try:
                sd.stop()
            except Exception:
                pass

        if persistent_output_path is not None:
            print(f"[MicrophoneRecorder Test] Saved WAV: {saved_path}")
            if metadata_path is not None:
                print(f"[MicrophoneRecorder Test] Saved metadata: {metadata_path}")
            return saved_path

        if saved_path is not None:
            saved_path.unlink(missing_ok=True)
        if metadata_path is not None:
            metadata_path.unlink(missing_ok=True)

    print("[MicrophoneRecorder Test] Completed and removed temporary WAV/JSON files.")
    return None


def test_microphone_recorder_record_and_playback_5s() -> None:
    """Backward-compatible 5s smoke test that does not persist output."""

    run_microphone_recorder_smoke_test()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Record from the microphone, optionally save the WAV/JSON output."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds.",
    )
    parser.add_argument(
        "--playback-seconds",
        type=float,
        default=5.0,
        help="How many seconds to play back after recording. Use 0 to disable playback.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Input sample rate in Hz.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of input channels.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist the recorded WAV/JSON files instead of deleting temp files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output WAV path. If provided, the recording will be kept.",
    )
    args = parser.parse_args()

    run_microphone_recorder_smoke_test(
        record_seconds=args.duration,
        playback_seconds=args.playback_seconds,
        sample_rate=args.sample_rate,
        channels=args.channels,
        save=bool(args.save or args.output is not None),
        output_path=args.output,
    )


if __name__ == "__main__":
    _main()
