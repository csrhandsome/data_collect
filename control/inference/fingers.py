"""Capture four unique RGB pairs; the GPU policy server encodes contact features."""

import threading
import uuid
from collections import deque

import numpy as np

from control.util.img_util import center_crop_and_resize_rgb_uint8


class FingerImages:
    def __init__(self, config, metadata):
        if metadata.get("finger_input_protocol") != "panda-finger-rgb-history-v1":
            raise ValueError("C server must support panda-finger-rgb-history-v1")
        self.history = deque(maxlen=4)
        self.baseline = None
        self.session_id = uuid.uuid4().hex
        self.max_skew = float(metadata["finger_max_skew_s"])
        self.max_age = float(metadata["finger_max_age_s"])
        self.hw = int(config["camera"].get("image_hw", 224))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.crop = float(config["camera"].get("crop_scale", 0.9))

    def start(self, arm):
        def poll():
            try:
                while not self.stop_event.is_set():
                    self.capture(arm)
                    self.stop_event.wait(0.005)
            except Exception as exc:
                self.error = exc

        self.thread = threading.Thread(target=poll, name="finger-rgb-history", daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def capture(self, arm):
        left, right, left_ns, right_ns = arm.gripper.get_tactile_frames()
        if left is None or right is None or left_ns is None or right_ns is None:
            return
        times = np.array([left_ns, right_ns], dtype=np.float64) / 1e9
        if abs(times[0] - times[1]) > self.max_skew:
            return
        with self.lock:
            latest = self.history[-1][2] if self.history else None
        if latest is not None and np.any(times <= latest):
            return
        images = tuple(
            center_crop_and_resize_rgb_uint8(image, out_hw=self.hw, crop_scale=self.crop)
            for image in (left, right)
        )
        with self.lock:
            if self.baseline is None:
                if arm.gripper.get_state().commanded_open_ratio < 0.95:
                    raise ValueError(
                        "C requires an open, no-contact gripper baseline before inference"
                    )
                self.baseline = images
            self.history.append((*images, times))

    def request_fields(self, robot_ts):
        if self.error is not None:
            raise RuntimeError("C finger camera capture failed") from self.error
        with self.lock:
            history = list(self.history)
            baseline = self.baseline
        if len(history) != 4:
            return None
        times = np.stack([row[2] for row in history])
        ages = robot_ts - times
        valid = ((ages >= -self.max_skew) & (ages <= self.max_age)).all(axis=1)
        if not valid.all():
            return None
        return {
            "observation.gripper_image_left": np.stack([row[0] for row in history]),
            "observation.gripper_image_right": np.stack([row[1] for row in history]),
            "finger_timestamps": times,
            "finger_valid_mask": valid,
            "finger_baseline_left": baseline[0].copy(),
            "finger_baseline_right": baseline[1].copy(),
            "finger_session_id": self.session_id,
        }
