from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Optional
import time
import wave

import numpy as np
import sounddevice as sd

from .util.audio_util import (
    _resolve_preferred_input_device,
    _resolve_supported_input_sample_rate,
    block_frames_for_sample_rate,
    default_microphone_output_path,
    play_audio,
    read_wav_pcm,
    save_staged_audio_recording,
)


def _microphone_worker_main(
    conn: Connection,
    sample_rate: int,
    channels: int,
    dtype_name: str,
    input_device: int | str | None,
    normalize_audio: bool,
    normalization_target_peak_ratio: float,
) -> None:
    dtype = np.dtype(dtype_name)
    requested_sample_rate = int(sample_rate)
    resolved_sample_rate = int(sample_rate)
    block_frames = block_frames_for_sample_rate(resolved_sample_rate)

    stream: sd.InputStream | None = None
    wave_writer: wave.Wave_write | None = None
    staged_wav_path: Optional[Path] = None
    staged_is_temp = False
    is_recording = False
    sample_count = 0

    chunk_timestamps_ns: list[int] = []
    input_buffer_adc_times: list[Optional[float]] = []
    input_overflow_timestamps_ns: list[int] = []
    audio_start_monotonic_ns: Optional[int] = None
    audio_stop_monotonic_ns: Optional[int] = None
    last_saved_path: Optional[Path] = None
    last_metadata_path: Optional[Path] = None
    last_error: Optional[str] = None

    def _summary() -> dict:
        return {
            "sample_rate": int(resolved_sample_rate),
            "audio_start_monotonic_ns": audio_start_monotonic_ns,
            "audio_stop_monotonic_ns": audio_stop_monotonic_ns,
            "last_saved_path": str(last_saved_path) if last_saved_path else None,
            "last_metadata_path": (
                str(last_metadata_path) if last_metadata_path else None
            ),
            "num_samples": int(sample_count),
            "num_input_overflows": len(input_overflow_timestamps_ns),
            "has_staged_audio": staged_wav_path is not None,
            "staged_is_temp": staged_is_temp,
        }

    def _close_stream() -> None:
        nonlocal stream
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        stream = None

    def _close_wave_writer() -> None:
        nonlocal wave_writer
        if wave_writer is None:
            return
        try:
            wave_writer.close()
        except Exception:
            pass
        wave_writer = None

    def _discard_staged_wav() -> None:
        nonlocal staged_wav_path
        nonlocal staged_is_temp
        if staged_wav_path is not None and staged_is_temp:
            try:
                staged_wav_path.unlink(missing_ok=True)
            except Exception:
                pass
        staged_wav_path = None
        staged_is_temp = False

    def _reset_state(*, keep_saved_paths: bool = False) -> None:
        nonlocal sample_count
        nonlocal audio_start_monotonic_ns
        nonlocal audio_stop_monotonic_ns
        nonlocal last_saved_path
        nonlocal last_metadata_path
        nonlocal last_error
        sample_count = 0
        chunk_timestamps_ns.clear()
        input_buffer_adc_times.clear()
        input_overflow_timestamps_ns.clear()
        audio_start_monotonic_ns = None
        audio_stop_monotonic_ns = None
        last_error = None
        _close_stream()
        _close_wave_writer()
        _discard_staged_wav()
        if not keep_saved_paths:
            last_saved_path = None
            last_metadata_path = None

    def _open_staged_wav() -> None:
        nonlocal staged_wav_path
        nonlocal staged_is_temp
        nonlocal wave_writer
        temp_file = NamedTemporaryFile(
            prefix="microphone_recorder_",
            suffix=".wav",
            delete=False,
        )
        temp_file.close()
        staged_wav_path = Path(temp_file.name)
        staged_is_temp = True
        wave_writer = wave.open(str(staged_wav_path), "wb")
        wave_writer.setnchannels(int(channels))
        wave_writer.setsampwidth(dtype.itemsize)
        wave_writer.setframerate(int(resolved_sample_rate))

    def _materialize_staged_audio(output_path_raw: str | None) -> dict:
        nonlocal staged_wav_path
        nonlocal staged_is_temp
        nonlocal last_saved_path
        nonlocal last_metadata_path

        if output_path_raw is None:
            last_saved_path = None
            last_metadata_path = None
            return {"ok": True, "saved_path": None, "metadata_path": None, **_summary()}

        if staged_wav_path is None or not staged_wav_path.is_file():
            return {
                "ok": False,
                "error": "No staged audio is available to save",
                **_summary(),
            }

        output_path = Path(output_path_raw)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            saved_path, metadata_path, _ = save_staged_audio_recording(
                staged_wav_path=staged_wav_path,
                staged_is_temp=staged_is_temp,
                output_path=output_path,
                sample_rate=resolved_sample_rate,
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
                num_samples=sample_count,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc), **_summary()}

        staged_wav_path = saved_path
        staged_is_temp = False
        last_saved_path = saved_path
        last_metadata_path = metadata_path

        return {
            "ok": True,
            "saved_path": str(saved_path),
            "metadata_path": str(last_metadata_path),
            **_summary(),
        }

    def _start_recording() -> dict:
        nonlocal resolved_sample_rate
        nonlocal block_frames
        nonlocal audio_start_monotonic_ns
        nonlocal stream

        _reset_state()
        try:
            resolved_sample_rate = _resolve_supported_input_sample_rate(
                input_device=input_device,
                requested_sample_rate=requested_sample_rate,
                channels=channels,
                dtype_name=dtype.name,
            )
            block_frames = block_frames_for_sample_rate(resolved_sample_rate)
            _open_staged_wav()
            stream_obj = sd.InputStream(
                samplerate=resolved_sample_rate,
                channels=channels,
                dtype=dtype.name,
                device=input_device,
                blocksize=block_frames,
                latency="high",
            )
            stream_obj.start()
        except Exception as exc:
            _close_stream()
            _close_wave_writer()
            _discard_staged_wav()
            return {"ok": False, "error": str(exc), **_summary()}

        stream = stream_obj
        return {"ok": True}

    try:
        while True:
            if is_recording:
                try:
                    assert stream is not None
                    assert wave_writer is not None
                    chunk, overflowed = stream.read(block_frames)
                    if chunk.size > 0:
                        chunk_arr = np.asarray(chunk, dtype=dtype)
                        wave_writer.writeframesraw(chunk_arr.tobytes())
                        sample_count += int(chunk_arr.shape[0])
                        chunk_timestamp_ns = time.monotonic_ns()
                        chunk_timestamps_ns.append(chunk_timestamp_ns)
                        stream_time = getattr(stream, "time", None)
                        if stream_time is None:
                            input_buffer_adc_times.append(None)
                        else:
                            try:
                                input_buffer_adc_times.append(float(stream_time))
                            except Exception:
                                input_buffer_adc_times.append(None)
                        if overflowed:
                            input_overflow_timestamps_ns.append(chunk_timestamp_ns)
                except Exception as exc:
                    last_error = str(exc)
                    is_recording = False
                    audio_stop_monotonic_ns = time.monotonic_ns()
                    _close_stream()
                    _close_wave_writer()

                while conn.poll():
                    command = conn.recv()
                    action = command.get("cmd")
                    if action == "stop":
                        is_recording = False
                        audio_stop_monotonic_ns = time.monotonic_ns()
                        _close_stream()
                        _close_wave_writer()
                        if last_error is not None:
                            conn.send(
                                {
                                    "ok": False,
                                    "error": f"Recording error: {last_error}",
                                    **_summary(),
                                }
                            )
                        else:
                            conn.send(_materialize_staged_audio(command.get("output_path")))
                    elif action == "shutdown":
                        is_recording = False
                        audio_stop_monotonic_ns = time.monotonic_ns()
                        _close_stream()
                        _close_wave_writer()
                        _discard_staged_wav()
                        conn.send({"ok": True})
                        return
                    elif action == "reset":
                        is_recording = False
                        audio_stop_monotonic_ns = time.monotonic_ns()
                        _reset_state()
                        conn.send({"ok": True, **_summary()})
                    elif action == "save":
                        conn.send(
                            {
                                "ok": False,
                                "error": "Cannot save recording while recording is active",
                                **_summary(),
                            }
                        )
                    elif action == "start":
                        conn.send(
                            {
                                "ok": False,
                                "error": "Recording is already active",
                                **_summary(),
                            }
                        )
                continue

            try:
                command = conn.recv()
            except EOFError:
                _close_stream()
                _close_wave_writer()
                _discard_staged_wav()
                return

            action = command.get("cmd")
            if action == "shutdown":
                _close_stream()
                _close_wave_writer()
                _discard_staged_wav()
                conn.send({"ok": True})
                return
            if action == "reset":
                _reset_state()
                conn.send({"ok": True, **_summary()})
                continue
            if action == "start":
                result = _start_recording()
                if result.get("ok"):
                    audio_start_monotonic_ns = time.monotonic_ns()
                    is_recording = True
                    last_error = None
                    conn.send({"ok": True, **_summary()})
                else:
                    last_error = str(result.get("error"))
                    conn.send({"ok": False, "error": last_error, **_summary()})
                continue
            if action == "stop":
                conn.send(_materialize_staged_audio(command.get("output_path")))
                continue
            if action == "save":
                conn.send(_materialize_staged_audio(command.get("output_path")))
                continue
            conn.send({"ok": False, "error": f"Unknown command: {action}"})
    finally:
        _close_stream()
        _close_wave_writer()
        _discard_staged_wav()


class MicrophoneRecorder:
    """Process-backed microphone recorder with episode-friendly WAV + JSON outputs."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: np.dtype | type[np.generic] = np.int16,
        default_output_path: Optional[Path] = None,
        input_device: int | str | None = None,
        normalize_audio: bool = False,
        normalization_target_peak_ratio: float = 0.95,
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
        self._input_device = (
            input_device
            if input_device is not None
            else _resolve_preferred_input_device()
        )
        self._normalize_audio = bool(normalize_audio)
        self._normalization_target_peak_ratio = float(normalization_target_peak_ratio)
        if not (0.0 < self._normalization_target_peak_ratio <= 1.0):
            raise ValueError("normalization_target_peak_ratio must be in (0, 1]")

        self._is_recording = False
        self._audio_start_monotonic_ns: Optional[int] = None
        self._audio_stop_monotonic_ns: Optional[int] = None
        self._last_saved_path: Optional[Path] = None
        self._last_metadata_path: Optional[Path] = None
        self._frames: tuple[np.ndarray, ...] = ()
        self._chunk_timestamps_ns: tuple[int, ...] = ()
        self._input_buffer_adc_times: tuple[Optional[float], ...] = ()
        self._input_overflow_timestamps_ns: tuple[int, ...] = ()

        self._ctx = mp.get_context("spawn")
        self._proc: Optional[mp.Process] = None
        self._parent_conn: Optional[Connection] = None
        self._child_conn: Optional[Connection] = None
        self._spawn_worker()

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
        return self._frames

    @property
    def chunk_timestamps_ns(self) -> tuple[int, ...]:
        return self._chunk_timestamps_ns

    @property
    def input_buffer_adc_times(self) -> tuple[Optional[float], ...]:
        return self._input_buffer_adc_times

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
    def input_device(self) -> int | str | None:
        return self._input_device

    @property
    def normalize_audio(self) -> bool:
        return self._normalize_audio

    @property
    def normalization_target_peak_ratio(self) -> float:
        return self._normalization_target_peak_ratio

    @property
    def last_saved_path(self) -> Optional[Path]:
        return self._last_saved_path

    @property
    def last_metadata_path(self) -> Optional[Path]:
        return self._last_metadata_path

    def _spawn_worker(self) -> None:
        self._parent_conn, self._child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_microphone_worker_main,
            args=(
                self._child_conn,
                int(self._sample_rate),
                int(self._channels),
                self._dtype.name,
                self._input_device,
                bool(self._normalize_audio),
                float(self._normalization_target_peak_ratio),
            ),
            daemon=True,
        )

    def start(self) -> None:
        if self._proc is None or (
            self._proc.exitcode is not None and not self._proc.is_alive()
        ):
            self._spawn_worker()
        if self._proc is None:
            raise RuntimeError("Microphone worker process is not initialized")
        if self._proc.is_alive():
            return
        self._proc.start()
        if self._child_conn is not None:
            self._child_conn.close()
            self._child_conn = None

    def stop(self) -> None:
        proc = self._proc
        conn = self._parent_conn
        if proc is not None and proc.is_alive() and conn is not None:
            try:
                conn.send({"cmd": "shutdown"})
                if conn.poll(2.0):
                    conn.recv()
            except Exception:
                pass
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        if self._parent_conn is not None:
            try:
                self._parent_conn.close()
            except Exception:
                pass
            self._parent_conn = None
        if self._child_conn is not None:
            try:
                self._child_conn.close()
            except Exception:
                pass
            self._child_conn = None
        self._proc = None
        self._is_recording = False

    def _ensure_worker_started(self) -> None:
        if self._proc is None or not self._proc.is_alive():
            self.start()

    def _clear_cached_metadata_lists(self) -> None:
        self._frames = ()
        self._chunk_timestamps_ns = ()
        self._input_buffer_adc_times = ()
        self._input_overflow_timestamps_ns = ()

    def _load_cached_metadata_lists(self) -> None:
        self._clear_cached_metadata_lists()
        if self._last_metadata_path is None or not self._last_metadata_path.is_file():
            return
        try:
            metadata = json.loads(self._last_metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._chunk_timestamps_ns = tuple(metadata.get("chunk_timestamps_ns", ()))
        self._input_buffer_adc_times = tuple(
            metadata.get("input_buffer_adc_times", ())
        )
        self._input_overflow_timestamps_ns = tuple(
            metadata.get("input_overflow_timestamps_ns", ())
        )

    def _sync_from_response(self, response: dict) -> None:
        sample_rate = response.get("sample_rate")
        if sample_rate is not None:
            self._sample_rate = int(sample_rate)
        self._audio_start_monotonic_ns = response.get("audio_start_monotonic_ns")
        self._audio_stop_monotonic_ns = response.get("audio_stop_monotonic_ns")
        saved_path = response.get("saved_path")
        self._last_saved_path = Path(saved_path) if saved_path else None
        metadata_path = response.get("metadata_path")
        self._last_metadata_path = Path(metadata_path) if metadata_path else None
        self._load_cached_metadata_lists()

    def _request(self, command: dict, *, timeout_s: float = 60.0) -> dict:
        self._ensure_worker_started()
        if self._parent_conn is None:
            raise RuntimeError("Microphone worker connection is not available")
        try:
            self._parent_conn.send(command)
            if not self._parent_conn.poll(timeout_s):
                raise RuntimeError(
                    f"Microphone worker did not respond to {command.get('cmd')}"
                )
            response = self._parent_conn.recv()
        except Exception:
            self.stop()
            raise
        self._sync_from_response(response)
        return response

    def reset(self) -> None:
        if self._is_recording:
            raise RuntimeError("Cannot reset while recording is active")
        self._clear_cached_metadata_lists()
        self._audio_start_monotonic_ns = None
        self._audio_stop_monotonic_ns = None
        self._last_saved_path = None
        self._last_metadata_path = None
        if self._proc is not None and self._proc.is_alive():
            self._request({"cmd": "reset"})

    def _prepare_input_settings(self) -> None:
        self._sample_rate = _resolve_supported_input_sample_rate(
            input_device=self._input_device,
            requested_sample_rate=self._sample_rate,
            channels=self._channels,
            dtype_name=self._dtype.name,
        )

    def _read_block_frames(self) -> int:
        return block_frames_for_sample_rate(self._sample_rate)

    def start_recording(self) -> bool:
        if self._is_recording:
            return False

        try:
            self.reset()
            response = self._request({"cmd": "start"})
        except Exception as exc:
            print(f"[MicrophoneRecorder] Failed to start recorder process: {exc}")
            return False

        if not response.get("ok", False):
            print(
                f"[MicrophoneRecorder] Input configuration error: {response.get('error')}"
            )
            return False

        self._is_recording = True
        return True

    def stop_recording(self) -> Optional[Path]:
        if not self._is_recording:
            return None

        response = self._request(
            {
                "cmd": "stop",
                "output_path": (
                    str(self._default_output_path)
                    if self._default_output_path is not None
                    else None
                ),
            }
        )
        self._is_recording = False
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "Unknown recording error")))
        return self._last_saved_path

    def save_recording(self, output_path: Path) -> Path:
        if self._is_recording:
            raise RuntimeError("Cannot save recording while recording is active")

        response = self._request({"cmd": "save", "output_path": str(Path(output_path))})
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "Failed to save recording")))
        if self._last_saved_path is None:
            raise RuntimeError("Recording did not produce a WAV file")
        return self._last_saved_path


def run_microphone_recorder_smoke_test(
    *,
    record_seconds: float = 5.0,
    playback_seconds: float = 5.0,
    sample_rate: int = 16000,
    channels: int = 1,
    save: bool = False,
    output_path: Optional[Path] = None,
    normalize_audio: bool = True,
    normalization_target_peak_ratio: float = 0.95,
) -> Optional[Path]:
    """Hardware smoke test: record from the mic, optionally save, then play back."""

    if record_seconds <= 0:
        raise ValueError("record_seconds must be > 0")
    if playback_seconds < 0:
        raise ValueError("playback_seconds must be >= 0")

    persistent_output_path = (
        Path(output_path)
        if output_path is not None
        else (default_microphone_output_path() if save else None)
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
            normalize_audio=normalize_audio,
            normalization_target_peak_ratio=normalization_target_peak_ratio,
        )
        saved_path: Optional[Path] = None
        metadata_path: Optional[Path] = None

        try:
            recorder.start()
            recorder._prepare_input_settings()

            save_mode = (
                "persistent save" if persistent_output_path is not None else "temp"
            )
            print(
                f"[MicrophoneRecorder Test] Recording for {record_seconds:.1f}s "
                f"to {resolved_output_path} ({save_mode}, {recorder.sample_rate} Hz)"
            )
            if not recorder.start_recording():
                raise RuntimeError("Failed to start microphone recording")

            time.sleep(record_seconds)
            saved_path = recorder.stop_recording()
            metadata_path = recorder.last_metadata_path
            if saved_path is None or not saved_path.is_file():
                raise RuntimeError("Recording did not produce a WAV file")

            if playback_seconds > 0:
                audio, saved_sample_rate = read_wav_pcm(
                    saved_path, require_nonempty=True
                )
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
                try:
                    recorder.stop_recording()
                except Exception:
                    pass
            recorder.stop()
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
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable peak normalization before saving the recorded WAV.",
    )
    parser.add_argument(
        "--normalization-target-peak-ratio",
        type=float,
        default=0.95,
        help="Peak normalization target as a fraction of full scale. Default: 0.95.",
    )
    args = parser.parse_args()

    run_microphone_recorder_smoke_test(
        record_seconds=args.duration,
        playback_seconds=args.playback_seconds,
        sample_rate=args.sample_rate,
        channels=args.channels,
        save=bool(args.save or args.output is not None),
        output_path=args.output,
        normalize_audio=args.normalize,
        normalization_target_peak_ratio=args.normalization_target_peak_ratio,
    )


if __name__ == "__main__":
    _main()
