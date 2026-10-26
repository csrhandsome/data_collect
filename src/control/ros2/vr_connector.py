"""Subscribe to VR state published by the separate Teleop node.

The publisher owns the HTTPS server. This process only keeps the latest sample,
matching the ``latest`` / ``stop`` surface used by collection.
"""

import json
from types import SimpleNamespace

from control.ros2.base_node import Ros2Node

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
    payload["pose_monotonic_ns"] = str(int(sample.pose_monotonic_ns))
    payload["raw_position"] = sample.raw_position
    payload["raw_quaternion_xyzw"] = sample.raw_quaternion_xyzw
    return json.dumps(payload)


def loads_vr(text):
    raw = json.loads(text)
    sample = {name: raw[name] for name in _FIELDS}
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
        super().__init__("vr", config.get("ros2", {}).get("namespace", ""))
        self._timeout = float(ros.get("startup_timeout_s", 10))
        self.start()

    def start(self):
        if self._source.node is not None:
            self._source.check()
            return
        try:
            self._sample = None
            super().start()
            self._source.wait(lambda: self._sample is not None, self._timeout)
        except BaseException:
            self.stop()
            raise

    def _setup(self, node):
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self._subscription = node.create_subscription(
            String,
            self._topic,
            self._receive,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )

    def _receive(self, message):
        try:
            sample = loads_vr(message.data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._source.node.get_logger().warning(f"Invalid VR sample: {exc}", throttle_duration_sec=2.0)
            return
        with self._source.condition:
            self._sample = sample
            self._source.condition.notify_all()

    def read(self):
        with self._source.condition:
            self._source.check()
            if self._sample is None:
                raise TimeoutError("No VR sample has been received")
            return self._sample

    @property
    def latest(self):
        return self.read()

    def close(self):
        super().close()
        self._sample = None

    def stop(self):
        self.close()
