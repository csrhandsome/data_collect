"""Subscribe to VR state published by the separate Teleop node.

The publisher owns the HTTPS server. This process only keeps the latest sample,
matching the ``latest`` / ``stop`` surface used by collection.
"""

import json
import logging
import time
from types import SimpleNamespace

from control.ros2.base_node import Ros2Node

logger = logging.getLogger(__name__)

_FIELDS = (
    "arm_enabled",
    "pos_x",
    "pos_y",
    "pos_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
    "x_pressed",
    "y_pressed",
    "save_pressed",
    "gripper_close",
    "gripper_open",
    "right_stick_x",
    "right_stick_y",
    "gripper_velocity_axis",
    "timestamp",
    "pose_seq",
)


def dumps_vr(sample):
    payload = {name: getattr(sample, name) for name in _FIELDS}
    payload["critical_pressed"] = bool(getattr(sample, "critical_pressed", False))
    payload["pose_monotonic_ns"] = str(int(sample.pose_monotonic_ns))
    payload["raw_position"] = sample.raw_position
    payload["raw_quaternion_xyzw"] = sample.raw_quaternion_xyzw
    return json.dumps(payload)


def loads_vr(text):
    raw = json.loads(text)
    sample = {name: raw[name] for name in _FIELDS}
    sample["critical_pressed"] = bool(raw.get("critical_pressed", False))
    sample["pose_monotonic_ns"] = int(raw["pose_monotonic_ns"])
    sample["pose_seq"] = int(sample["pose_seq"])
    for key in ("raw_position", "raw_quaternion_xyzw"):
        value = raw.get(key)
        sample[key] = tuple(value) if value is not None else None
    return SimpleNamespace(**sample)


class Ros2VR(Ros2Node):
    def __init__(self, config):
        cfg = config.get("vr", {})
        ros = cfg.get("ros2", {})
        self._topic = ros.get("topic", "/vr/input")
        self._sample = None
        self._last_missing_error_ns = 0
        self._stale_after_s = float(cfg.get("stale_after_s", 0.5))
        super().__init__(
            "vr", config.get("ros2", {}).get("namespace", ""),
            monitoring=config.get("ros2", {}).get("monitoring"),
        )
        self._timeout = float(ros.get("startup_timeout_s", 10))
        self.start()

    def start(self):
        if self._source.node is not None:
            self._source.check()
            return
        try:
            self._sample = None
            super().start()
            try:
                self._source.wait(lambda: self._sample is not None, self._timeout)
            except TimeoutError:
                logger.error(
                    "VR topic %s unavailable after %.1fs; continuing with idle fallback",
                    self._topic,
                    self._timeout,
                )
        except BaseException:
            self.stop()
            raise

    def _setup(self, node):
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self._subscription = self._source.subscribe(
            String,
            self._topic,
            self._receive,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            expected_hz=100.0, timeout_s=0.5,
        )

    def _receive(self, message):
        try:
            sample = loads_vr(message.data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._source.node.get_logger().warning(f"Invalid VR sample: {exc}", throttle_duration_sec=2.0)
            return False
        with self._source.condition:
            self._sample = sample
            self._source.condition.notify_all()
        # The ROS publisher can keep sending while the headset itself stops updating.
        return sample.pose_monotonic_ns > 0 and (
            -0.1 <= (time.monotonic_ns() - sample.pose_monotonic_ns) / 1e9 <= self._stale_after_s
        )

    def read(self):
        with self._source.condition:
            self._source.check()
            if self._sample is None:
                now = time.monotonic_ns()
                if now - self._last_missing_error_ns >= 2_000_000_000:
                    self._last_missing_error_ns = now
                    logger.error("No VR sample received on %s; continuing with idle fallback", self._topic)
                return SimpleNamespace(
                    arm_enabled=False,
                    pos_x=0.0,
                    pos_y=0.0,
                    pos_z=0.0,
                    quat_x=0.0,
                    quat_y=0.0,
                    quat_z=0.0,
                    quat_w=1.0,
                    x_pressed=False,
                    y_pressed=False,
                    save_pressed=False,
                    gripper_close=False,
                    gripper_open=False,
                    right_stick_x=0.0,
                    right_stick_y=0.0,
                    gripper_velocity_axis=0.0,
                    timestamp=0.0,
                    pose_seq=0,
                    critical_pressed=False,
                    pose_monotonic_ns=0,
                    raw_position=None,
                    raw_quaternion_xyzw=None,
                )
            return self._sample

    @property
    def latest(self):
        return self.read()

    def close(self):
        super().close()
        self._sample = None

    def stop(self):
        self.close()
