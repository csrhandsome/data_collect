"""
Dual camera managers for DROID-style deployment.

支持两类双相机：
- DualRealsenseManager: 双 RealSense
- DualRGBCameraManager: 双普通 RGB 相机(OpenCV)
"""

from __future__ import annotations

import glob
from typing import Optional, Tuple

import numpy as np

try:
    from control.camera_connector import RGBCameraConnector, RealSenseConnector
except ModuleNotFoundError:
    from camera_connector import RGBCameraConnector, RealSenseConnector


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
            align_depth_to_color=True,
        )

        print(
            f"[DualCameraManager] 初始化腕部相机 (serial: {wrist_serial or 'auto'})..."
        )
        self.wrist_camera = RealSenseConnector(
            serial=wrist_serial,
            width=width,
            height=height,
            fps=fps,
            align_depth_to_color=True,
        )

        self._connected = False

    def connect(self) -> "DualRealsenseManager":
        if self._connected:
            return self

        print("[DualCameraManager] 连接外部相机...")
        self.external_camera.connect()

        print("[DualCameraManager] 连接腕部相机...")
        self.wrist_camera.connect()

        self._connected = True
        print("[DualCameraManager] 双 RealSense 系统已连接")
        return self

    def update(self, timeout_ms: int = 1000) -> Tuple[bool, bool]:
        external_ok = self.external_camera.update(timeout=timeout_ms)
        wrist_ok = self.wrist_camera.update(timeout=timeout_ms)
        return external_ok, wrist_ok

    def get_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.external_camera.img, self.wrist_camera.img

    def get_depths(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.external_camera.depth, self.wrist_camera.depth

    def close(self) -> None:
        print("[DualCameraManager] 关闭相机...")
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

    def connect(self) -> "DualRGBCameraManager":
        if self._connected:
            return self

        print("[DualRGBCameraManager] 连接外部相机...")
        self.external_camera.connect()

        print("[DualRGBCameraManager] 连接腕部相机...")
        self.wrist_camera.connect()

        self._connected = True
        print("[DualRGBCameraManager] 双 RGB 相机系统已连接")
        return self

    def update(self, timeout_ms: int = 1000) -> Tuple[bool, bool]:
        external_ok = self.external_camera.update(timeout=timeout_ms)
        wrist_ok = self.wrist_camera.update(timeout=timeout_ms)
        return external_ok, wrist_ok

    def get_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.external_camera.img, self.wrist_camera.img

    def get_depths(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return None, None

    def close(self) -> None:
        print("[DualRGBCameraManager] 关闭相机...")
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
    parser.add_argument("--list-devices", action="store_true", help="列出所有相机设备")
    parser.add_argument(
        "--max-rgb-devices", type=int, default=10, help="RGB 相机扫描最大索引"
    )
    parser.add_argument("--width", type=int, default=640, help="图像宽度")
    parser.add_argument("--height", type=int, default=480, help="图像高度")
    parser.add_argument("--fps", type=int, default=30, help="目标帧率")
    parser.add_argument(
        "--frames", type=int, default=None, help="采集帧数（不指定则持续显示）"
    )
    parser.add_argument("--no-display", action="store_true", help="不显示图像窗口")
    args = parser.parse_args()

    if args.list_devices:
        if args.camera_type == "realsense":
            print("\n可用的 RealSense 设备：")
            devices = list_realsense_devices()
            if not devices:
                print("  未找到设备")
            for i, dev in enumerate(devices):
                print(f"  [{i}] {dev['name']} (序列号: {dev['serial']})")
        else:
            print("\n可用的 RGB 相机：")
            devices = list_rgb_devices(max_index=args.max_rgb_devices)
            if not devices:
                print("  未找到设备")
            for i, dev in enumerate(devices):
                print(
                    f"  [{i}] index={dev['index']} path={dev['path']} "
                    f"size=({dev['width']}, {dev['height']}) fps={dev['fps']:.2f}"
                )
        return

    if args.camera_type == "realsense":
        if args.external_serial is None or args.wrist_serial is None:
            devices = list_realsense_devices()
            if len(devices) < 2:
                print(
                    f"[错误] 需要至少 2 个 RealSense 相机，但只找到 {len(devices)} 个"
                )
                return

            if args.external_serial is None:
                args.external_serial = devices[0]["serial"]
                print(
                    f"[自动选择] 外部相机: {devices[0]['name']} ({args.external_serial})"
                )

            if args.wrist_serial is None:
                args.wrist_serial = devices[1]["serial"]
                print(
                    f"[自动选择] 腕部相机: {devices[1]['name']} ({args.wrist_serial})"
                )

        manager = DualRealsenseManager(
            external_serial=args.external_serial,
            wrist_serial=args.wrist_serial,
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

    with manager:
        if not args.no_display:
            cv2.namedWindow("External Camera", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Wrist Camera", cv2.WINDOW_NORMAL)
            print("\n按 'q' 或 ESC 退出，按 's' 保存当前帧")

        frame_count = 0
        max_frames = args.frames if args.frames else float("inf")
        print(f"\n开始采集{'持续' if args.frames is None else f'{args.frames} 帧'}...")

        while frame_count < max_frames:
            external_ok, wrist_ok = manager.update(timeout_ms=1000)

            if external_ok and wrist_ok:
                external_img, wrist_img = manager.get_images()
                if external_img is not None and wrist_img is not None:
                    print(
                        f"[{frame_count:3d}] ✓ External: {external_img.shape}, Wrist: {wrist_img.shape}"
                    )

                    if not args.no_display:
                        external_bgr = cv2.cvtColor(external_img, cv2.COLOR_RGB2BGR)
                        wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)

                        cv2.imshow("External Camera", external_bgr)
                        cv2.imshow("Wrist Camera", wrist_bgr)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:
                            print("\n用户退出")
                            break
                        if key == ord("s"):
                            cv2.imwrite(f"external_{frame_count:04d}.png", external_bgr)
                            cv2.imwrite(f"wrist_{frame_count:04d}.png", wrist_bgr)
                            print(
                                f"  已保存: external_{frame_count:04d}.png, wrist_{frame_count:04d}.png"
                            )
                else:
                    print(f"[{frame_count:3d}] ✗ 图像为 None")
            else:
                print(
                    f"[{frame_count:3d}] ✗ External: {external_ok}, Wrist: {wrist_ok}"
                )

            frame_count += 1

        if not args.no_display:
            cv2.destroyAllWindows()

    print("\n测试完成")


if __name__ == "__main__":
    main()
