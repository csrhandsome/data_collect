"""
Dual RealSense Camera Manager for DROID-style deployment

管理两个 RealSense 相机（外部相机 + 腕部相机），模拟 DROID 的相机配置
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from realsense_connector import RealSenseConnector


class DualCameraManager:
    """
    双相机管理器，用于 DROID 风格的部署

    相机配置：
    - external_camera: 外部固定相机（模拟 DROID 的 exterior_image_1_left）
    - wrist_camera: 腕部相机（模拟 DROID 的 wrist_image_left）
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
        """
        初始化双相机管理器

        Args:
            external_serial: 外部相机序列号（如果为 None，使用第一个发现的相机）
            wrist_serial: 腕部相机序列号（如果为 None，使用第二个发现的相机）
            width: 相机分辨率宽度
            height: 相机分辨率高度
            fps: 帧率
        """
        print("[DualCameraManager] 初始化双相机系统...")

        # 外部相机
        print(f"[DualCameraManager] 初始化外部相机 (serial: {external_serial or 'auto'})...")
        self.external_camera = RealSenseConnector(
            serial=external_serial,
            width=width,
            height=height,
            fps=fps,
            align_depth_to_color=True,
        )

        # 腕部相机
        print(f"[DualCameraManager] 初始化腕部相机 (serial: {wrist_serial or 'auto'})...")
        self.wrist_camera = RealSenseConnector(
            serial=wrist_serial,
            width=width,
            height=height,
            fps=fps,
            align_depth_to_color=True,
        )

        self._connected = False

    def connect(self) -> "DualCameraManager":
        """连接所有相机"""
        if self._connected:
            return self

        print("[DualCameraManager] 连接外部相机...")
        self.external_camera.connect()

        print("[DualCameraManager] 连接腕部相机...")
        self.wrist_camera.connect()

        self._connected = True
        print("[DualCameraManager] 双相机系统已连接")
        return self

    def update(self, timeout_ms: int = 1000) -> Tuple[bool, bool]:
        """
        更新所有相机

        Returns:
            (external_ok, wrist_ok): 两个相机的更新状态
        """
        external_ok = self.external_camera.update(timeout=timeout_ms)
        wrist_ok = self.wrist_camera.update(timeout=timeout_ms)
        return external_ok, wrist_ok

    def get_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取两个相机的图像

        Returns:
            (external_img, wrist_img): RGB 图像 (H, W, 3) uint8，如果失败则为 None
        """
        external_img = self.external_camera.img
        wrist_img = self.wrist_camera.img
        return external_img, wrist_img

    def get_depths(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取两个相机的深度图

        Returns:
            (external_depth, wrist_depth): 深度图 (H, W) float32 (单位：米)
        """
        external_depth = self.external_camera.depth
        wrist_depth = self.wrist_camera.depth
        return external_depth, wrist_depth

    def close(self) -> None:
        """关闭所有相机"""
        print("[DualCameraManager] 关闭相机...")
        self.external_camera.close()
        self.wrist_camera.close()
        self._connected = False

    def __enter__(self) -> "DualCameraManager":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


def list_realsense_devices() -> list[dict]:
    """
    列出所有已连接的 RealSense 设备

    Returns:
        设备信息列表，每个设备包含 {'serial': str, 'name': str}
    """
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


def main():
    """测试双相机系统"""
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description="测试双 RealSense 相机")
    parser.add_argument("--external-serial", type=str, default=None, help="外部相机序列号")
    parser.add_argument("--wrist-serial", type=str, default=None, help="腕部相机序列号")
    parser.add_argument("--list-devices", action="store_true", help="列出所有相机设备")
    parser.add_argument("--frames", type=int, default=None, help="采集帧数（不指定则持续显示）")
    parser.add_argument("--no-display", action="store_true", help="不显示图像窗口")
    args = parser.parse_args()

    if args.list_devices:
        print("\n可用的 RealSense 设备：")
        devices = list_realsense_devices()
        if not devices:
            print("  未找到设备")
        for i, dev in enumerate(devices):
            print(f"  [{i}] {dev['name']} (序列号: {dev['serial']})")
        return

    # 自动分配相机序列号
    if args.external_serial is None or args.wrist_serial is None:
        devices = list_realsense_devices()
        if len(devices) < 2:
            print(f"[错误] 需要至少 2 个 RealSense 相机，但只找到 {len(devices)} 个")
            return

        if args.external_serial is None:
            args.external_serial = devices[0]["serial"]
            print(f"[自动选择] 外部相机: {devices[0]['name']} ({args.external_serial})")

        if args.wrist_serial is None:
            args.wrist_serial = devices[1]["serial"]
            print(f"[自动选择] 腕部相机: {devices[1]['name']} ({args.wrist_serial})")

    # 创建双相机管理器
    manager = DualCameraManager(
        external_serial=args.external_serial,
        wrist_serial=args.wrist_serial,
    )

    with manager:
        if not args.no_display:
            cv2.namedWindow("External Camera", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Wrist Camera", cv2.WINDOW_NORMAL)
            print("\n按 'q' 或 ESC 退出，按 's' 保存当前帧")

        frame_count = 0
        max_frames = args.frames if args.frames else float('inf')

        print(f"\n开始采集{'持续' if args.frames is None else f'{args.frames} 帧'}...")

        while frame_count < max_frames:
            external_ok, wrist_ok = manager.update(timeout_ms=1000)

            if external_ok and wrist_ok:
                external_img, wrist_img = manager.get_images()

                if external_img is not None and wrist_img is not None:
                    print(f"[{frame_count:3d}] ✓ External: {external_img.shape}, Wrist: {wrist_img.shape}")

                    if not args.no_display:
                        # 转换 RGB 到 BGR 用于 OpenCV 显示
                        external_bgr = cv2.cvtColor(external_img, cv2.COLOR_RGB2BGR)
                        wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)

                        cv2.imshow("External Camera", external_bgr)
                        cv2.imshow("Wrist Camera", wrist_bgr)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' 或 ESC
                            print("\n用户退出")
                            break
                        elif key == ord('s'):  # 保存图像
                            cv2.imwrite(f"external_{frame_count:04d}.png", external_bgr)
                            cv2.imwrite(f"wrist_{frame_count:04d}.png", wrist_bgr)
                            print(f"  已保存: external_{frame_count:04d}.png, wrist_{frame_count:04d}.png")
                else:
                    print(f"[{frame_count:3d}] ✗ 图像为 None")
            else:
                print(f"[{frame_count:3d}] ✗ External: {external_ok}, Wrist: {wrist_ok}")

            frame_count += 1

        if not args.no_display:
            cv2.destroyAllWindows()

    print("\n测试完成")


if __name__ == "__main__":
    main()
