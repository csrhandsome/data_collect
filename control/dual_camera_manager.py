"""
Dual camera managers for DROID-style deployment.

支持两类双相机：
- DualRealsenseManager: 双 RealSense
- DualRGBCameraManager: 双普通 RGB 相机(OpenCV)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import glob
import threading
import time
from typing import Optional, Tuple
import zlib

import numpy as np

try:
    from control.camera_connector import RGBCameraConnector, RealSenseConnector
    from control.util.img_util import center_crop_and_resize_rgb_uint8
except ModuleNotFoundError:
    from camera_connector import RGBCameraConnector, RealSenseConnector
    from control.util.img_util import center_crop_and_resize_rgb_uint8


@dataclass(frozen=True)
class CameraFrameTimestamps:
    camera_timestamp: float
    host_capture_monotonic_ns: int


class DualRealsenseManager:
    """
    双 RealSense 相机管理器，用于 DROID 风格的部署。

    相机配置：
    - external_camera: 外部固定相机
    - wrist_camera: 腕部相机
    """

    def __init__(
        self,
        *,
        external_serial: Optional[str] = None,
        wrist_serial: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        align_depth_to_color: bool = True,
        background_poll: bool = False,
        background_poll_interval_s: float = 0.0,
        background_timeout_ms: int = 1000,
        crop_scale: float = 0.9,
        out_hw: Optional[int] = None,
    ) -> None:
        print("[DualCameraManager] 初始化双 RealSense 系统...")

        print(
            f"[DualCameraManager] 初始化外部相机 (serial: {external_serial or 'auto'})..."
        )
        self.external_camera = RealSenseConnector(
            serial=external_serial,
            width=width,
            height=height,
            fps=fps,
            enable_depth=enable_depth,
            align_depth_to_color=align_depth_to_color,
        )

        print(
            f"[DualCameraManager] 初始化腕部相机 (serial: {wrist_serial or 'auto'})..."
        )
        self.wrist_camera = RealSenseConnector(
            serial=wrist_serial,
            width=width,
            height=height,
            fps=fps,
            enable_depth=enable_depth,
            align_depth_to_color=align_depth_to_color,
        )

        self._connected = False
        self._background_poll = bool(background_poll)
        self._background_poll_interval_s = max(float(background_poll_interval_s), 0.0)
        self._background_timeout_ms = int(background_timeout_ms)
        self._crop_scale = float(crop_scale)
        self._out_hw = int(out_hw) if out_hw is not None else None
        self._image_lock = threading.Lock()
        self._background_stop_event = threading.Event()
        self._background_thread: Optional[threading.Thread] = None
        self._background_error: Optional[Exception] = None
        self._external_ok = False
        self._wrist_ok = False
        self._external_img: Optional[np.ndarray] = None
        self._wrist_img: Optional[np.ndarray] = None
        self._external_depth: Optional[np.ndarray] = None
        self._wrist_depth: Optional[np.ndarray] = None
        self._external_timestamp: Optional[CameraFrameTimestamps] = None
        self._wrist_timestamp: Optional[CameraFrameTimestamps] = None

    def _connect_cameras(self) -> None:
        if self._connected:
            return

        print("[DualCameraManager] 连接外部相机...")
        self.external_camera.connect()

        print("[DualCameraManager] 连接腕部相机...")
        self.wrist_camera.connect()

        self._connected = True
        self._background_error = None
        print("[DualCameraManager] 双 RealSense 系统已连接")

    def _prepare_image(self, rgb: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if rgb is None:
            return None
        rgb = np.asarray(rgb, dtype=np.uint8)
        if self._out_hw is None:
            return rgb
        return center_crop_and_resize_rgb_uint8(
            rgb,
            crop_scale=self._crop_scale,
            out_hw=self._out_hw,
        )

    def _cache_frames(
        self,
        external_ok: bool,
        wrist_ok: bool,
        external_host_capture_monotonic_ns: Optional[int] = None,
        wrist_host_capture_monotonic_ns: Optional[int] = None,
    ) -> None:
        external_img = self._prepare_image(self.external_camera.img)
        wrist_img = self._prepare_image(self.wrist_camera.img)
        external_depth = self.external_camera.depth
        wrist_depth = self.wrist_camera.depth
        external_camera_timestamp = float(self.external_camera.timestamp)
        wrist_camera_timestamp = float(self.wrist_camera.timestamp)
        with self._image_lock:
            self._external_ok = bool(external_ok)
            self._wrist_ok = bool(wrist_ok)
            if external_img is not None:
                self._external_img = external_img
                if external_host_capture_monotonic_ns is not None:
                    self._external_timestamp = CameraFrameTimestamps(
                        camera_timestamp=external_camera_timestamp,
                        host_capture_monotonic_ns=int(
                            external_host_capture_monotonic_ns
                        ),
                    )
            if wrist_img is not None:
                self._wrist_img = wrist_img
                if wrist_host_capture_monotonic_ns is not None:
                    self._wrist_timestamp = CameraFrameTimestamps(
                        camera_timestamp=wrist_camera_timestamp,
                        host_capture_monotonic_ns=int(wrist_host_capture_monotonic_ns),
                    )
            if external_depth is not None:
                self._external_depth = external_depth
            if wrist_depth is not None:
                self._wrist_depth = wrist_depth

    def _update_once(self, timeout_ms: int) -> Tuple[bool, bool]:
        external_ok = self.external_camera.update(timeout=timeout_ms)
        external_host_capture_monotonic_ns = time.monotonic_ns()
        wrist_ok = self.wrist_camera.update(timeout=timeout_ms)
        wrist_host_capture_monotonic_ns = time.monotonic_ns()
        self._cache_frames(
            external_ok,
            wrist_ok,
            external_host_capture_monotonic_ns=external_host_capture_monotonic_ns,
            wrist_host_capture_monotonic_ns=wrist_host_capture_monotonic_ns,
        )
        return external_ok, wrist_ok

    def _background_worker(self) -> None:
        try:
            self._connect_cameras()
        except Exception as exc:
            self._background_error = exc
            return

        while not self._background_stop_event.is_set():
            try:
                self._update_once(timeout_ms=self._background_timeout_ms)
                self._background_error = None
            except Exception as exc:
                self._background_error = exc
            if self._background_stop_event.wait(self._background_poll_interval_s):
                break

    def _start_background_thread(self) -> None:
        if self._background_thread is not None and self._background_thread.is_alive():
            return

        self._background_stop_event.clear()
        self._background_thread = threading.Thread(
            target=self._background_worker,
            name="dual-realsense-manager",
            daemon=True,
        )
        self._background_thread.start()

    def connect(self) -> "DualRealsenseManager":
        if self._background_poll:
            self._start_background_thread()
            return self

        self._connect_cameras()
        return self

    def wait_for_frames(self, timeout_s: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        if not self._connected and not self._background_poll:
            self._connect_cameras()

        start = time.time()
        while True:
            if self._background_error is not None:
                raise RuntimeError(
                    f"DualRealsenseManager background polling failed: {self._background_error}"
                ) from self._background_error

            if not self._background_poll:
                self._update_once(timeout_ms=self._background_timeout_ms)

            external_img, wrist_img = self.get_images()
            if external_img is not None and wrist_img is not None:
                return external_img, wrist_img

            if timeout_s > 0 and (time.time() - start) > timeout_s:
                raise RuntimeError(
                    f"Camera timeout: Failed to get both frames after {timeout_s:.1f} seconds."
                )
            time.sleep(0.01)

    def update(self, timeout_ms: int = 1000) -> Tuple[bool, bool]:
        if self._background_poll:
            with self._image_lock:
                return self._external_ok, self._wrist_ok

        if not self._connected:
            self._connect_cameras()
        return self._update_once(timeout_ms=timeout_ms)

    def get_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._image_lock:
            return self._external_img, self._wrist_img

    def get_frames(
        self,
    ) -> tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[CameraFrameTimestamps],
        Optional[CameraFrameTimestamps],
    ]:
        with self._image_lock:
            return (
                self._external_img,
                self._wrist_img,
                self._external_timestamp,
                self._wrist_timestamp,
            )

    def get_depths(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._image_lock:
            return self._external_depth, self._wrist_depth

    @property
    def background_error(self) -> Optional[Exception]:
        return self._background_error

    def close(self) -> None:
        print("[DualCameraManager] 关闭相机...")
        self._background_stop_event.set()
        if self._background_thread is not None:
            self._background_thread.join(
                timeout=max(self._background_timeout_ms / 1000.0, 1.0)
            )
            self._background_thread = None
        self.external_camera.close()
        self.wrist_camera.close()
        self._connected = False

    def __enter__(self) -> "DualRealsenseManager":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


class DualRGBCameraManager:
    """
    双普通 RGB 相机管理器，底层使用 OpenCV VideoCapture。

    相机配置：
    - right_camera_device: 右侧相机
    - left_camera_device: 左侧相机
    """

    def __init__(
        self,
        *,
        right_camera_device: int | str = 1,
        left_camera_device: int | str = 2,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: Optional[int] = None,
        background_poll: bool = False,
        background_poll_interval_s: float = 0.01,
        background_timeout_ms: int = 1000,
    ) -> None:
        print("[DualRGBCameraManager] 初始化双 RGB 相机系统...")

        print(
            f"[DualRGBCameraManager] 初始化外部相机 (device: {right_camera_device})..."
        )
        self.external_camera = RGBCameraConnector(
            device=right_camera_device,
            width=width,
            height=height,
            fps=fps,
            backend=backend,
        )

        print(
            f"[DualRGBCameraManager] 初始化腕部相机 (device: {left_camera_device})..."
        )
        self.wrist_camera = RGBCameraConnector(
            device=left_camera_device,
            width=width,
            height=height,
            fps=fps,
            backend=backend,
        )

        self._connected = False
        self._background_poll = bool(background_poll)
        self._background_poll_interval_s = max(float(background_poll_interval_s), 0.0)
        self._background_timeout_ms = int(background_timeout_ms)
        self._image_lock = threading.Lock()
        self._background_stop_event = threading.Event()
        self._background_thread: Optional[threading.Thread] = None
        self._background_error: Optional[Exception] = None
        self._external_ok = False
        self._wrist_ok = False
        self._external_img: Optional[np.ndarray] = None
        self._wrist_img: Optional[np.ndarray] = None

    def _connect_cameras(self) -> None:
        if self._connected:
            return

        print("[DualRGBCameraManager] 连接外部相机...")
        self.external_camera.connect()

        print("[DualRGBCameraManager] 连接腕部相机...")
        self.wrist_camera.connect()

        self._connected = True
        self._background_error = None
        print("[DualRGBCameraManager] 双 RGB 相机系统已连接")

    def _cache_images(self, external_ok: bool, wrist_ok: bool) -> None:
        external_img = self.external_camera.img
        wrist_img = self.wrist_camera.img
        with self._image_lock:
            self._external_ok = bool(external_ok)
            self._wrist_ok = bool(wrist_ok)
            if external_img is not None:
                self._external_img = external_img
            if wrist_img is not None:
                self._wrist_img = wrist_img

    def _update_once(self, timeout_ms: int) -> Tuple[bool, bool]:
        external_ok = self.external_camera.update(timeout=timeout_ms)
        wrist_ok = self.wrist_camera.update(timeout=timeout_ms)
        self._cache_images(external_ok, wrist_ok)
        return external_ok, wrist_ok

    def _background_worker(self) -> None:
        try:
            self._connect_cameras()
        except Exception as exc:
            self._background_error = exc
            return

        while not self._background_stop_event.is_set():
            try:
                self._update_once(timeout_ms=self._background_timeout_ms)
                self._background_error = None
            except Exception as exc:
                self._background_error = exc
            if self._background_stop_event.wait(self._background_poll_interval_s):
                break

    def _start_background_thread(self) -> None:
        if self._background_thread is not None and self._background_thread.is_alive():
            return

        self._background_stop_event.clear()
        self._background_thread = threading.Thread(
            target=self._background_worker,
            name="dual-rgb-camera-manager",
            daemon=True,
        )
        self._background_thread.start()

    def connect(self) -> "DualRGBCameraManager":
        if self._background_poll:
            self._start_background_thread()
            return self

        self._connect_cameras()
        return self

    def update(self, timeout_ms: int = 1000) -> Tuple[bool, bool]:
        if self._background_poll:
            with self._image_lock:
                return self._external_ok, self._wrist_ok

        if not self._connected:
            self._connect_cameras()
        return self._update_once(timeout_ms=timeout_ms)

    def get_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._image_lock:
            return self._external_img, self._wrist_img

    def get_depths(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return None, None

    @property
    def background_error(self) -> Optional[Exception]:
        return self._background_error

    def close(self) -> None:
        print("[DualRGBCameraManager] 关闭相机...")
        self._background_stop_event.set()
        if self._background_thread is not None:
            self._background_thread.join(
                timeout=max(self._background_timeout_ms / 1000.0, 1.0)
            )
            self._background_thread = None
        self.external_camera.close()
        self.wrist_camera.close()
        self._connected = False

    def __enter__(self) -> "DualRGBCameraManager":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


DualCameraManager = DualRealsenseManager
DuralRGBCameraManager = DualRGBCameraManager


def list_realsense_devices() -> list[dict]:
    """列出所有已连接的 RealSense 设备。"""
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[Error] pyrealsense2 未安装，无法列出设备")
        return []

    context = rs.context()
    devices = []
    for device in context.query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        devices.append({"serial": serial, "name": name})
    return devices


def list_rgb_devices(max_index: int = 10) -> list[dict]:
    """列出可能可用的普通 RGB 相机。"""
    try:
        import cv2
    except ImportError:
        print("[Error] opencv-python 未安装，无法列出 RGB 相机")
        return []

    linux_paths = {path for path in sorted(glob.glob("/dev/video*"))}
    devices = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index)
        try:
            if not capture.isOpened():
                continue
            width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
            height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            path = f"/dev/video{index}" if f"/dev/video{index}" in linux_paths else None
            devices.append(
                {
                    "index": index,
                    "path": path,
                    "width": width,
                    "height": height,
                    "fps": fps,
                }
            )
        finally:
            capture.release()

    if not devices:
        for path in sorted(linux_paths):
            devices.append(
                {"index": None, "path": path, "width": 0, "height": 0, "fps": 0.0}
            )
    return devices


def parse_camera_device(value: str) -> int | str:
    value = value.strip()
    if value.lstrip("-").isdigit():
        return int(value)
    return value


class _FPSWindow:
    """Track recent per-stream frame rate over a short sliding window."""

    def __init__(self, window_s: float = 1.5) -> None:
        self._window_s = max(float(window_s), 0.2)
        self._timestamps: deque[float] = deque()
        self._fps = 0.0

    def tick(self, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else float(now)
        self._timestamps.append(now)
        cutoff = now - self._window_s
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

        if len(self._timestamps) >= 2:
            duration = max(self._timestamps[-1] - self._timestamps[0], 1e-6)
            self._fps = (len(self._timestamps) - 1) / duration
        else:
            self._fps = 0.0
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


class _FrameStreamStats:
    """Track read FPS and approximate fresh-frame FPS using frame fingerprints."""

    def __init__(self, window_s: float = 1.5, sample_step: int = 16) -> None:
        self._read_meter = _FPSWindow(window_s=window_s)
        self._fresh_meter = _FPSWindow(window_s=window_s)
        self._sample_step = max(int(sample_step), 1)
        self._last_signature: Optional[int] = None

    def _signature(self, frame: np.ndarray) -> int:
        sampled = np.ascontiguousarray(
            frame[:: self._sample_step, :: self._sample_step]
        )
        return int(zlib.crc32(sampled.tobytes()) & 0xFFFFFFFF)

    def tick(
        self,
        *,
        frame: Optional[np.ndarray],
        ok: bool,
        now: Optional[float] = None,
    ) -> tuple[float, float, bool]:
        now = time.monotonic() if now is None else float(now)
        if not ok or frame is None:
            return self.read_fps, self.fresh_fps, False

        self._read_meter.tick(now)

        signature = self._signature(frame)
        is_fresh = signature != self._last_signature
        if is_fresh:
            self._last_signature = signature
            self._fresh_meter.tick(now)

        return self.read_fps, self.fresh_fps, is_fresh

    @property
    def read_fps(self) -> float:
        return self._read_meter.fps

    @property
    def fresh_fps(self) -> float:
        return self._fresh_meter.fps


def main():
    """测试双相机系统。"""
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description="测试双相机系统（RealSense / RGB）")
    parser.add_argument(
        "--camera-type",
        choices=["realsense", "rgb"],
        default="rgb",
        help="测试哪一类双相机管理器",
    )
    parser.add_argument(
        "--external-serial", type=str, default=None, help="外部 RealSense 序列号"
    )
    parser.add_argument(
        "--wrist-serial", type=str, default=None, help="腕部 RealSense 序列号"
    )
    parser.add_argument(
        "--external-device",
        type=str,
        default="1",
        help="外部 RGB 相机设备，支持索引或设备路径",
    )
    parser.add_argument(
        "--wrist-device",
        type=str,
        default="2",
        help="腕部 RGB 相机设备，支持索引或设备路径",
    )
    parser.add_argument("--width", type=int, default=640, help="图像宽度")
    parser.add_argument("--height", type=int, default=480, help="图像高度")
    parser.add_argument(
        "--fps",
        type=int,
        default=120,
        help="目标帧率；RGB 相机默认通过 OpenCV 尝试设置为 120 FPS",
    )
    args = parser.parse_args()

    if args.camera_type == "realsense":
        external_serial = args.external_serial
        wrist_serial = args.wrist_serial
        if external_serial is None or wrist_serial is None:
            devices = list_realsense_devices()
            if len(devices) < 2:
                raise RuntimeError(
                    f"需要至少 2 个 RealSense 相机，但只找到 {len(devices)} 个"
                )
            external_serial = external_serial or devices[0]["serial"]
            wrist_serial = wrist_serial or devices[1]["serial"]

        manager = DualRealsenseManager(
            external_serial=external_serial,
            wrist_serial=wrist_serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    else:
        manager = DualRGBCameraManager(
            right_camera_device=parse_camera_device(args.external_device),
            left_camera_device=parse_camera_device(args.wrist_device),
            width=args.width,
            height=args.height,
            fps=args.fps,
        )

    def _draw_overlay(
        frame_bgr: np.ndarray,
        *,
        camera_name: str,
        read_fps: float,
        fresh_fps: float,
        reported_fps: Optional[float],
        ok: bool,
    ) -> np.ndarray:
        overlay = frame_bgr.copy()
        lines = [
            camera_name,
            f"read fps: {read_fps:5.1f}",
            f"fresh fps: {fresh_fps:5.1f}",
            (
                f"reported fps: {reported_fps:5.1f}"
                if reported_fps is not None
                else "reported fps: n/a"
            ),
            f"status: {'OK' if ok else 'NO FRAME'}",
        ]

        y = 30
        for line in lines:
            cv2.putText(
                overlay,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if ok else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28
        return overlay

    with manager:
        cv2.namedWindow("External Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Wrist Camera", cv2.WINDOW_NORMAL)
        print(f"\n实时显示中，按 'q' 或 ESC 退出。当前请求帧率: {args.fps} FPS")
        print(
            "窗口中的 `read fps` 是读取频率，`fresh fps` 是基于帧变化估算的新帧频率。"
        )

        external_stats = _FrameStreamStats()
        wrist_stats = _FrameStreamStats()
        last_report_at = 0.0

        try:
            while True:
                external_ok, wrist_ok = manager.update(timeout_ms=1000)
                loop_now = time.monotonic()

                external_img, wrist_img = manager.get_images()
                if external_img is None or wrist_img is None:
                    continue

                external_read_fps, external_fresh_fps, _ = external_stats.tick(
                    frame=external_img,
                    ok=external_ok,
                    now=loop_now,
                )
                wrist_read_fps, wrist_fresh_fps, _ = wrist_stats.tick(
                    frame=wrist_img,
                    ok=wrist_ok,
                    now=loop_now,
                )

                external_reported_fps = None
                wrist_reported_fps = None
                if hasattr(manager, "external_camera") and hasattr(
                    manager.external_camera, "fps"
                ):
                    external_reported_fps = float(manager.external_camera.fps)
                if hasattr(manager, "wrist_camera") and hasattr(
                    manager.wrist_camera, "fps"
                ):
                    wrist_reported_fps = float(manager.wrist_camera.fps)

                external_bgr = cv2.cvtColor(external_img, cv2.COLOR_RGB2BGR)
                wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)

                external_bgr = _draw_overlay(
                    external_bgr,
                    camera_name="External Camera",
                    read_fps=external_read_fps,
                    fresh_fps=external_fresh_fps,
                    reported_fps=external_reported_fps,
                    ok=external_ok,
                )
                wrist_bgr = _draw_overlay(
                    wrist_bgr,
                    camera_name="Wrist Camera",
                    read_fps=wrist_read_fps,
                    fresh_fps=wrist_fresh_fps,
                    reported_fps=wrist_reported_fps,
                    ok=wrist_ok,
                )

                cv2.imshow("External Camera", external_bgr)
                cv2.imshow("Wrist Camera", wrist_bgr)

                if loop_now - last_report_at >= 1.0:
                    print(
                        "[FPS] "
                        f"external read={external_read_fps:5.1f}, "
                        f"fresh={external_fresh_fps:5.1f}, "
                        f"reported={external_reported_fps if external_reported_fps is not None else float('nan'):5.1f} | "
                        f"wrist read={wrist_read_fps:5.1f}, "
                        f"fresh={wrist_fresh_fps:5.1f}, "
                        f"reported={wrist_reported_fps if wrist_reported_fps is not None else float('nan'):5.1f}"
                    )
                    last_report_at = loop_now

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
