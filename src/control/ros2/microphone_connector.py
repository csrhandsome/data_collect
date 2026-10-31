"""ROS PCM16 audio subscription with streamed WAV and episode metadata output.

Input is std_msgs/Int16MultiArray, interleaved at the configured sample rate
and channel count. It has no source stamp; metadata records host receipt times.
"""

import json
import shutil
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from control.ros2.subscription import Subscription


class Ros2Microphone:
    def __init__(self, config):
        cfg = config.get("audio", {})
        ros = cfg.get("ros2", {})
        self.sample_rate = int(cfg.get("sample_rate", 16000))
        self.channels = int(cfg.get("channels", 1))
        self._timeout = float(ros.get("startup_timeout_s", 10))
        self._max_age = float(ros.get("max_age_s", 0.5))
        if (not np.isfinite([self.sample_rate, self.channels, self._timeout, self._max_age]).all()
                or min(self.sample_rate, self.channels, self._timeout, self._max_age) <= 0):
            raise ValueError("Audio sample rate, channels and max age must be positive")
        self._topic = ros.get("topic", "/microphone/audio")
        self._source = Subscription(
            "microphone", config.get("ros2", {}).get("namespace", ""),
            monitoring=config.get("ros2", {}).get("monitoring"),
        )
        self._last_received_ns = None
        self._wave = self._temp = None
        self.default_output_path = self.last_metadata_path = None
        self.audio_start_monotonic_ns = self.audio_stop_monotonic_ns = None
        self.is_recording = False

    def start(self):
        if self._source.node is not None:
            return
        try:
            self._last_received_ns = None
            self._source.start(self._setup)
            self._source.wait(lambda: self._last_received_ns is not None, self._timeout)
        except BaseException:
            self.stop()
            raise

    def _setup(self, node):
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Int16MultiArray

        self._subscription = self._source.subscribe(
            Int16MultiArray, self._topic, self._receive,
            QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT),
            timeout_s=self._max_age,
        )

    def _receive(self, message):
        pcm = np.asarray(message.data, dtype="<i2")
        if not pcm.size or pcm.size % self.channels or message.layout.data_offset:
            self._source.node.get_logger().warning("Invalid interleaved PCM16 audio chunk",
                                                   throttle_duration_sec=2.0)
            return False
        now = time.monotonic_ns()
        with self._source.condition:
            if self.is_recording:
                if now - self._last_received_ns > self._max_age * 1e9:
                    self._recording_error = "ROS microphone stream has a gap"
                try:
                    self._wave.writeframesraw(pcm.tobytes())
                    self._samples += pcm.size // self.channels
                    self._chunks.append(now)
                except Exception as exc:
                    self._recording_error = str(exc)
            self._last_received_ns = now
            self._source.condition.notify_all()

    def start_recording(self):
        with self._source.condition:
            self._source.check()
            if self.is_recording:
                return False
            if self._last_received_ns is None or (
                time.monotonic_ns() - self._last_received_ns > self._max_age * 1e9
            ):
                raise TimeoutError("ROS microphone data is stale")
            self._temp = tempfile.TemporaryDirectory(prefix="ros2-audio-")
            self._staged = Path(self._temp.name) / "episode.wav"
            try:
                self._wave = wave.open(str(self._staged), "wb")
                self._wave.setnchannels(self.channels)
                self._wave.setsampwidth(2)
                self._wave.setframerate(self.sample_rate)
            except BaseException:
                self._discard()
                raise
            self._chunks, self._samples, self._recording_error = [], 0, None
            self.last_metadata_path = None
            self.audio_start_monotonic_ns = time.monotonic_ns()
            self.audio_stop_monotonic_ns = None
            self.is_recording = True
            return True

    def stop_recording(self):
        with self._source.condition:
            if not self.is_recording:
                return None
            self.is_recording = False
            self.audio_stop_monotonic_ns = time.monotonic_ns()
            try:
                self._wave.close()
                self._wave = None
                if self.default_output_path is None:
                    return None
                self._source.check()
                if self._recording_error:
                    raise RuntimeError(self._recording_error)
                if not self._samples or (
                    self.audio_stop_monotonic_ns - self._last_received_ns > self._max_age * 1e9
                ):
                    raise TimeoutError("No fresh ROS microphone audio was recorded")
                path = Path(self.default_output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self._staged, path)
                self.last_metadata_path = path.with_suffix(".audio.json")
                self.last_metadata_path.write_text(json.dumps({
                    "audio_path": str(path), "sample_rate": self.sample_rate,
                    "channels": self.channels, "dtype": "int16", "source": "ros2",
                    "topic": self._topic, "timestamp_source": "host_receipt",
                    "audio_start_monotonic_ns": self.audio_start_monotonic_ns,
                    "audio_stop_monotonic_ns": self.audio_stop_monotonic_ns,
                    "num_samples": self._samples, "num_chunks": len(self._chunks),
                    "chunk_timestamps_ns": self._chunks,
                    "input_buffer_adc_times": [None] * len(self._chunks),
                }, indent=2))
                return path
            finally:
                self._discard()

    def _discard(self):
        if self._wave is not None:
            self._wave.close()
            self._wave = None
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None

    def stop(self):
        self.default_output_path = None
        try:
            self.stop_recording()
        finally:
            self._source.close()
