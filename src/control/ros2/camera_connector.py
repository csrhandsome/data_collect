"""Subscribe to two raw Image streams and expose the common camera-pair API."""

import time

import numpy as np

from control.ros2.base_node import Ros2Node
from control.types import CameraPair
from control.util.img_util import center_crop_and_resize_rgb_uint8


def image_rgb(message):
    """Decode supported 8-bit encodings, including rows with padding."""
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    count = channels.get(message.encoding)
    if count is None:
        raise ValueError(f"Unsupported ROS image encoding: {message.encoding}")
    height, width, step = message.height, message.width, message.step
    if height <= 0 or width <= 0 or step < width * count:
        raise ValueError("Invalid ROS image dimensions or row stride")
    data = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if data.size != height * step:
        raise ValueError("ROS image payload does not match height * step")
    rgb = data.reshape(height, step)[:, :width * count].reshape(height, width, count)
    if count == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    else:
        rgb = rgb[:, :, :3]
        if message.encoding.startswith("bgr"):
            rgb = rgb[:, :, ::-1]
    return np.ascontiguousarray(rgb)


class Ros2Cameras(Ros2Node[CameraPair]):
    def __init__(self, config):
        cfg = config.get("camera", {})
        ros = cfg.get("ros2", {})
        self._max_age = float(cfg.get("max_age_s", 0.2))
        self._crop = float(cfg.get("crop_scale", 0.9))
        self._hw = int(cfg.get("image_hw", 224))
        self._raw = {}
        self._cache = {}
        self._counts = {"front": 0, "wrist": 0}
        self._expected_hz = float(cfg.get("fps", 30))
        super().__init__(
            "camera", config.get("ros2", {}).get("namespace", ""), background=False,
            monitoring=config.get("ros2", {}).get("monitoring"),
        )
        self._timeout = float(cfg.get("startup_timeout_s", 10))
        self._topics = {
            "front": ros.get("external_image_topic", "/camera/external/color/image_raw"),
            "wrist": ros.get("wrist_image_topic", "/camera/wrist/color/image_raw"),
        }
        self.start()

    def start(self):
        if self._source.node is not None:
            self._source.check()
            return
        try:
            # Pump on the caller thread. A second rclpy spin thread stalls this
            # subscription after the control loop and the other ROS nodes are up.
            super().start()
            self._wait_ready(self._timeout)
        except BaseException:
            self.close()
            raise

    def _wait_ready(self, timeout):
        self._source.wait(self._ready, timeout)

    def _setup(self, node):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self._subscriptions = [
            self._source.subscribe(
                Image, topic, lambda msg, key=key: self._receive(key, msg), qos_profile_sensor_data,
                expected_hz=self._expected_hz, timeout_s=self._max_age,
            )
            for key, topic in self._topics.items()
        ]

    def _receive(self, key, message):
        stamp = message.header.stamp
        source_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if source_ns <= 0 or message.height <= 0 or message.width <= 0:
            self._source.node.get_logger().warning(
                "ROS camera message has no header stamp or image size", throttle_duration_sec=2.0,
            )
            return False
        # Keep the latest packet only. Cropping here makes the control thread
        # miss the other camera while it resizes a backlog.
        packet = (
            bytes(message.data), int(message.height), int(message.width), int(message.step),
            str(message.encoding), time.monotonic_ns(), source_ns,
        )
        self._counts[key] = self._counts.get(key, 0) + 1
        self._raw[key] = packet

    def _ready(self):
        if len(self._raw) != 2:
            return False
        now = time.monotonic_ns()
        return max(now - self._raw["front"][5], now - self._raw["wrist"][5]) <= self._max_age * 1e9

    def _resized(self, key):
        data, height, width, step, encoding, capture_ns, source_ns = self._raw[key]
        cached = self._cache.get(key)
        if cached is not None and cached[0] == source_ns:
            return cached[1], capture_ns, source_ns
        message = type("Image", (), {})()
        message.data = data
        message.height = height
        message.width = width
        message.step = step
        message.encoding = encoding
        rgb = center_crop_and_resize_rgb_uint8(
            image_rgb(message), crop_scale=self._crop, out_hw=self._hw, use_tensorflow=False,
        )
        self._cache[key] = (source_ns, rgb)
        return rgb, capture_ns, source_ns

    def read(self):
        # Drain ready callbacks even when the cached pair is still fresh.
        self._source.poll(0.0)
        # A USB camera can pause longer than max_age. Wait for the next pair
        # instead of ending collection on one gap.
        deadline = time.monotonic() + max(self._max_age, 1.0)
        while not self._ready():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                now = time.monotonic_ns()
                ages = {
                    key: round((now - value[5]) / 1e9, 3) for key, value in self._raw.items()
                }
                raise TimeoutError(
                    "ROS camera pair is stale or exceeds maximum timestamp skew "
                    f"ages={ages} counts={dict(self._counts)}"
                )
            self._source.poll(min(0.05, remaining))
        front = self._resized("front")
        wrist = self._resized("wrist")
        return CameraPair(front[0], wrist[0], front[1], wrist[1], front[2] / 1e9, wrist[2] / 1e9)

    def get_frames(self):
        return self.read()

    def close(self):
        super().close()
        self._raw.clear()
        self._cache.clear()
