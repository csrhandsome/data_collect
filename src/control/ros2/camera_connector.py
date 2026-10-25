"""Subscribe to two raw Image streams and expose the common camera-pair API."""

import time

import numpy as np

from control.ros2.subscription import Subscription
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


class Ros2Cameras:
    def __init__(self, config):
        cfg = config.get("camera", {})
        ros = cfg.get("ros2", {})
        self._max_age = float(cfg.get("max_age_s", 0.2))
        self._max_skew = float(ros.get("max_pair_skew_s", 0.1))
        self._crop = float(cfg.get("crop_scale", 0.9))
        self._hw = int(cfg.get("image_hw", 224))
        self._frames = {}
        self._source = Subscription("camera", config.get("ros2", {}).get("namespace", ""))
        self._topics = {
            "front": ros.get("external_image_topic", "/camera/external/color/image_raw"),
            "wrist": ros.get("wrist_image_topic", "/camera/wrist/color/image_raw"),
        }
        try:
            self._source.start(self._setup)
            self._source.wait(self._ready, float(cfg.get("startup_timeout_s", 10)))
        except BaseException:
            self.close()
            raise

    def _setup(self, node):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self._subscriptions = [
            node.create_subscription(Image, topic, lambda msg, key=key: self._receive(key, msg),
                                     qos_profile_sensor_data)
            for key, topic in self._topics.items()
        ]

    def _receive(self, key, message):
        try:
            stamp = message.header.stamp
            source_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            age_ns = self._source.node.get_clock().now().nanoseconds - source_ns
            if source_ns <= 0 or not -0.1e9 <= age_ns <= self._max_age * 1e9:
                raise ValueError("ROS camera source stamp is stale or clocks are unsynchronized")
            rgb = center_crop_and_resize_rgb_uint8(
                image_rgb(message), crop_scale=self._crop, out_hw=self._hw,
            )
            capture_ns = time.monotonic_ns() - max(0, age_ns)
        except (ValueError, TypeError) as exc:
            self._source.node.get_logger().warning(str(exc), throttle_duration_sec=2.0)
            return
        with self._source.condition:
            previous = self._frames.get(key)
            if previous is not None and source_ns <= previous[2]:
                return
            self._frames[key] = (rgb, capture_ns, source_ns)
            self._source.condition.notify_all()

    def _ready(self):
        if len(self._frames) != 2:
            return False
        front, wrist = self._frames["front"], self._frames["wrist"]
        now = time.monotonic_ns()
        return (max(now - front[1], now - wrist[1]) <= self._max_age * 1e9
                and abs(front[2] - wrist[2]) <= self._max_skew * 1e9)

    def get_frames(self):
        with self._source.condition:
            self._source.check()
            if not self._ready():
                raise TimeoutError("ROS camera pair is stale or exceeds maximum timestamp skew")
            front, wrist = self._frames["front"], self._frames["wrist"]
            return CameraPair(front[0], wrist[0], front[1], wrist[1], front[2] / 1e9, wrist[2] / 1e9)

    def close(self):
        self._source.close()
