"""Configured camera/microphone factories; hardware imports remain lazy."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from control.base_node import BaseNode
from control.types import CameraPair


class FakeCameras(BaseNode[CameraPair]):
    def __init__(self, config):
        self.config = config
        self.start_ns = time.monotonic_ns()

    def start(self):
        pass

    def read(self):
        return self.get_frames()

    def get_frames(self):
        cfg = self.config.get("camera", {})
        period = round(1e9 / float(cfg.get("fps", 30)))
        ns = self.start_ns + (time.monotonic_ns() - self.start_ns) // period * period
        hw = int(cfg.get("image_hw", 224))
        image = np.full((hw, hw, 3), (ns // period) % 255, dtype=np.uint8)
        return CameraPair(image, image.copy(), ns, ns, ns / 1e6, ns / 1e6)

    def close(self):
        pass


class RealCameras(BaseNode[CameraPair | None]):
    def __init__(self, config):
        from control.direct.dual_camera_manager import DualRealsenseManager

        cfg = config.get("camera", {})
        self.manager = DualRealsenseManager(
            external_serial=cfg.get("external_camera_serial"),
            wrist_serial=cfg.get("wrist_camera_serial"),
            width=int(cfg.get("width", 640)),
            height=int(cfg.get("height", 480)),
            fps=int(cfg.get("fps", 30)),
            enable_depth=not cfg.get("color_only", True),
            background_poll=True,
            crop_scale=float(cfg.get("crop_scale", 0.9)),
            out_hw=int(cfg.get("image_hw", 224)),
        )
        self._timeout = float(cfg.get("startup_timeout_s", 10))
        self._started = False
        self.start()

    def start(self):
        if self._started:
            return
        try:
            self.manager.connect()
            self.manager.wait_for_frames(timeout_s=self._timeout)
            self._started = True
        except BaseException:
            self.manager.close()
            raise

    def read(self):
        return self.get_frames()

    def get_frames(self):
        front, wrist, ft, wt = self.manager.get_frames()
        if any(x is None for x in (front, wrist, ft, wt)):
            return None
        return CameraPair(
            front,
            wrist,
            ft.host_capture_monotonic_ns,
            wt.host_capture_monotonic_ns,
            ft.camera_timestamp,
            wt.camera_timestamp,
        )

    def close(self):
        self.manager.close()
        self._started = False


def make_cameras(config, dry_run=False):
    if dry_run:
        return FakeCameras(config)
    if device_backend(config, "camera") == "ros2":
        from control.ros2.camera_connector import Ros2Cameras

        return Ros2Cameras(config)
    return RealCameras(config)


def device_backend(config, section):
    default = "ros2" if config.get("robot", {}).get("backend", "panda") == "ros2" else "direct"
    name = config.get(section, {}).get("backend", default)
    if name not in ("direct", "ros2"):
        raise ValueError(f"{section}.backend must be direct or ros2")
    return name


class FakeVR(BaseNode):
    def __init__(self):
        self.start_ns = time.monotonic_ns()

    def start(self):
        pass

    def read(self):
        return self.latest

    @property
    def latest(self):
        ns = time.monotonic_ns()
        return SimpleNamespace(
            arm_enabled=True,
            pos_x=(ns - self.start_ns) / 1e9 * 0.01,
            pos_y=0.0,
            pos_z=0.0,
            quat_x=0.0,
            quat_y=0.0,
            quat_z=0.0,
            quat_w=1.0,
            save_pressed=False,
            x_pressed=False,
            y_pressed=False,
            gripper_close=False,
            gripper_open=False,
            gripper_velocity_axis=0.0,
            pose_monotonic_ns=ns,
            pose_seq=1 + (ns - self.start_ns) // 11_111_111,
        )

    def stop(self):
        pass

    def close(self):
        self.stop()


def make_vr(config, dry_run=False):
    if dry_run:
        return FakeVR()
    if device_backend(config, "vr") == "ros2":
        from control.ros2.vr_connector import Ros2VR

        return Ros2VR(config)
    from control.collection.controller_hints import build_controller_hints
    from control.vr_input import VRInputProcess

    cfg = config.get("vr", {})
    reader = VRInputProcess(
        host=cfg.get("host", "0.0.0.0"),
        port=int(cfg.get("port", 4443)),
        long_press_s=float(cfg.get("long_press_s", 0.5)),
        position_alpha=float(cfg.get("position_alpha", 0.6)),
        rotation_alpha=float(cfg.get("rotation_alpha", 0.35)),
        controller_hints=build_controller_hints(config),
    )
    reader.start()
    return reader


def make_microphone(config, dry_run=False):
    if not config.get("audio", {}).get("enabled", False):
        return None
    if dry_run:
        from control.collection.audio import SyntheticMicrophone

        return SyntheticMicrophone(config.get("audio", {}))
    if device_backend(config, "audio") == "ros2":
        from control.ros2.microphone_connector import Ros2Microphone

        microphone = Ros2Microphone(config)
        microphone.start()
        return microphone
    from control.direct.microphone_connector import MicrophoneRecorder

    cfg = config["audio"]
    microphone = MicrophoneRecorder(
        sample_rate=int(cfg.get("sample_rate", 16000)),
        channels=int(cfg.get("channels", 1)),
        input_device=cfg.get("input_device"),
    )
    microphone.start()
    return microphone
