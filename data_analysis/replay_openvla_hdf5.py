#!/usr/bin/env python3
"""
重放（Replay）之前记录的机械臂轨迹

使用说明:
1. 自动查找最新的记录: python replay_demo.py
2. 指定HDF5文件: python replay_demo.py --file path/to/file.h5
3. 调整播放速度: python replay_demo.py --speed 0.5
4. 循环播放: python replay_demo.py --loop 3

数据来源:
- manual_control 记录的 HDF5 文件
- 位于 data/hdf5/ 目录下
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np


def _default_hdf5_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "hdf5"


def find_latest_log(log_dir: str | Path) -> str:
    """查找最新的日志文件"""
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"日志目录不存在: {log_dir}")

    h5_files = list(log_path.glob("gamepad_control_*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"在 {log_dir} 中未找到任何 gamepad_control_*.h5 文件")

    latest_file = max(h5_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


def load_trajectory(file_path: str | Path) -> dict:
    """从HDF5文件加载轨迹数据"""
    print(f"正在加载轨迹: {file_path}")

    with h5py.File(file_path, "r") as f:
        print(f"可用数据集: {list(f.keys())}")

        if "action" in f:
            actions = np.array(f["action"], dtype=np.float32)
            hz = float(f.attrs.get("control_frequency_hz", 100.0))
            print(f"动作序列长度: {len(actions)} 个样本, 采样频率: {hz:.1f} Hz")

            trajectory_data = {"action": actions, "control_frequency_hz": hz}

            if "observation" in f and "image_primary" in f["observation"]:
                images = np.array(f["observation/image_primary"], dtype=np.uint8)
                trajectory_data["images"] = images
                print(f"✓ 加载了 {len(images)} 帧图像数据")
            else:
                print("⚠️ 该文件没有图像数据")

            if "initial_position" in f.attrs:
                trajectory_data["initial_position"] = list(f.attrs["initial_position"])
            if "initial_orientation" in f.attrs:
                trajectory_data["initial_orientation"] = list(f.attrs["initial_orientation"])

            return trajectory_data

        q = np.array(f["q"])
        dq = np.array(f["dq"]) if "dq" in f else np.zeros_like(q)
        time_array = np.array(f["time"]) if "time" in f else None
        print(f"轨迹长度: {len(q)} 个样本")
        if time_array is not None and len(time_array) > 1:
            duration = float(time_array[-1] - time_array[0])
            if duration > 0:
                print(f"轨迹时长: {duration:.2f} 秒")
                print(f"采样频率: {len(q) / duration:.1f} Hz")
        return {"q": q, "dq": dq, "time": time_array}


def _slice_recorded_images(
    trajectory: dict,
    start_index: int,
    end_index: int | None,
) -> np.ndarray | None:
    images = trajectory.get("images")
    if images is None:
        return None

    images = np.asarray(images, dtype=np.uint8)
    if images.size == 0:
        return None

    if "action" not in trajectory:
        return images[start_index:end_index]

    actions = np.asarray(trajectory["action"], dtype=np.float32)
    stop = len(actions) if end_index is None else end_index
    sliced_actions = actions[start_index:stop]
    sliced_images = images[start_index:stop]
    if len(sliced_images) != len(sliced_actions):
        print(
            f"[Display] 警告: 图像数量 ({len(sliced_images)}) 与动作数量 ({len(sliced_actions)}) 不匹配，跳过图像显示"
        )
        return None
    return sliced_images


class _ReplayDisplayManager:
    def __init__(
        self,
        *,
        recorded_images: np.ndarray | None,
        camera,
        show_images: bool,
        show_live_camera: bool,
    ) -> None:
        self.recorded_images = recorded_images
        self.camera = camera
        self.show_images = bool(show_images)
        self.live_camera_requested = bool(show_live_camera)
        self.show_live_camera = bool(show_live_camera and camera is not None)
        self.cv2 = None
        self.enabled = False
        self.live_camera_disabled = False
        self.live_camera_fail_count = 0

    @property
    def has_images(self) -> bool:
        return self.recorded_images is not None and len(self.recorded_images) > 0

    def setup(self) -> None:
        if self.live_camera_requested and self.camera is None:
            print("[Display] 警告: 未提供 camera，忽略实时相机显示")

        if not self.show_images and not self.show_live_camera:
            print("[Display] 不显示图像")
            return

        try:
            import cv2
        except ImportError as e:
            print(f"\n[Error] OpenCV 导入失败: {e}")
            print("安装: pip install opencv-python")
            return
        except Exception as e:
            print(f"\n[Error] 初始化显示失败: {e}")
            return

        self.cv2 = cv2
        self.enabled = True
        print(f"[Display] OpenCV version: {cv2.__version__}")

        if self.show_live_camera:
            print("[Display] 预热相机...")
            for _ in range(5):
                self.camera.update(timeout=200)
                time.sleep(0.1)
            print("[Display] 相机预热完成")

        if self.show_images and self.has_images:
            print(f"[Display] 将显示采集的图像: {len(self.recorded_images)} 帧")
            if self.show_live_camera:
                self._open_window("Replay - Recorded vs Live", 1280, 480)
                print("[Display] 图像对比模式：左=采集图像，右=实时相机")
            else:
                self._open_window("Replay - Recorded Images", 640, 480)
                print("[Display] 显示采集时保存的图像")
        elif self.show_live_camera:
            self._open_window("Replay - Live Camera", 640, 480)
            print("[Display] 显示实时相机画面")
        else:
            print(
                f"[Display] 警告: show_images={self.show_images}, has_images={self.has_images}, show_live_camera={self.show_live_camera}"
            )
            self.enabled = False
            return

        print("提示：按 'q' 键关闭窗口（不会停止回放）")

    def _open_window(self, name: str, width: int, height: int) -> None:
        self.cv2.namedWindow(name, self.cv2.WINDOW_NORMAL)
        self.cv2.resizeWindow(name, width, height)

    def _get_recorded_frame(self, frame_index: int) -> np.ndarray | None:
        if not self.show_images or not self.has_images:
            return None
        if frame_index >= len(self.recorded_images):
            if frame_index == 0:
                print(
                    f"[Display] 无法获取采集图像: frame_index={frame_index}, images={len(self.recorded_images)}"
                )
            return None
        recorded_img_bgr = self.cv2.cvtColor(
            self.recorded_images[frame_index].copy(), self.cv2.COLOR_RGB2BGR
        )
        if frame_index == 0:
            print(
                f"[Display] 第一帧采集图像: shape={recorded_img_bgr.shape}, dtype={recorded_img_bgr.dtype}"
            )
        return recorded_img_bgr

    def _get_live_frame(self, frame_index: int) -> np.ndarray | None:
        if not self.show_live_camera or self.live_camera_disabled:
            return None

        ok = self.camera.update(timeout=50)
        if ok and self.camera.img is not None:
            live_img_bgr = self.cv2.cvtColor(self.camera.img.copy(), self.cv2.COLOR_RGB2BGR)
            if frame_index == 0:
                print(
                    f"[Display] 第一帧实时图像: shape={live_img_bgr.shape}, dtype={live_img_bgr.dtype}"
                )
            self.live_camera_fail_count = 0
            return live_img_bgr

        self.live_camera_fail_count += 1
        if frame_index == 0:
            print(f"[Display] 无法获取实时图像: ok={ok}")

        if self.live_camera_fail_count >= 10 and not self.live_camera_disabled:
            print(
                f"\n[Display] 实时相机连续失败 {self.live_camera_fail_count} 次，自动切换为只显示采集图像"
            )
            self.live_camera_disabled = True
            self.cv2.destroyAllWindows()
            if self.show_images and self.has_images:
                self._open_window("Replay - Recorded Images", 640, 480)
            else:
                self.enabled = False
        return None

    def render_step(self, step_info: dict) -> None:
        if not self.enabled or self.cv2 is None:
            return

        frame_index = int(step_info["frame_index"])
        frame_count = int(step_info["frame_count"])
        loop_index = int(step_info["loop_index"])
        loop_count = int(step_info["loop_count"])
        gripper_open = bool(step_info["gripper_open"])

        try:
            recorded_img_bgr = self._get_recorded_frame(frame_index)
            live_img_bgr = self._get_live_frame(frame_index)
            display_img = None
            window_name = "Replay"

            if recorded_img_bgr is not None and live_img_bgr is not None:
                height = min(recorded_img_bgr.shape[0], live_img_bgr.shape[0])
                recorded_resized = self.cv2.resize(
                    recorded_img_bgr,
                    (
                        int(recorded_img_bgr.shape[1] * height / recorded_img_bgr.shape[0]),
                        height,
                    ),
                )
                live_resized = self.cv2.resize(
                    live_img_bgr,
                    (
                        int(live_img_bgr.shape[1] * height / live_img_bgr.shape[0]),
                        height,
                    ),
                )
                self.cv2.putText(
                    recorded_resized,
                    "RECORDED",
                    (10, 30),
                    self.cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                self.cv2.putText(
                    live_resized,
                    "LIVE",
                    (10, 30),
                    self.cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )
                display_img = np.hstack([recorded_resized, live_resized])
                window_name = "Replay - Recorded vs Live"
                if frame_index == 0:
                    print(f"[Display] 对比模式: 合并图像 shape={display_img.shape}")
            elif recorded_img_bgr is not None:
                display_img = recorded_img_bgr
                window_name = "Replay - Recorded Images"
                if frame_index == 0:
                    print(f"[Display] 单图模式: 采集图像 shape={display_img.shape}")
            elif live_img_bgr is not None:
                display_img = live_img_bgr
                window_name = "Replay - Live Camera"
                if frame_index == 0:
                    print(f"[Display] 单图模式: 实时图像 shape={display_img.shape}")

            if display_img is None:
                if frame_index == 0:
                    print("[Display] 错误: 无法生成显示图像")
                return

            progress = (frame_index / frame_count) * 100 if frame_count > 0 else 0.0
            info_text = [
                f"Frame: {frame_index + 1}/{frame_count}",
                f"Progress: {progress:.1f}%",
                f"Loop: {loop_index + 1}/{loop_count}",
                f"Gripper: {'Open' if gripper_open else 'Closed'}",
            ]

            y_offset = display_img.shape[0] - 140
            for text in info_text:
                self.cv2.putText(
                    display_img,
                    text,
                    (10, y_offset),
                    self.cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                y_offset += 35

            self.cv2.imshow(window_name, display_img)
            if frame_index == 0:
                print(f"[Display] 第一帧已显示在窗口: {window_name}")

            key = self.cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[Display] 用户按 'q' 键，关闭图像显示（回放继续）")
                self.cv2.destroyAllWindows()
                self.enabled = False
        except Exception as e:
            if frame_index == 0 or frame_index % 100 == 0:
                print(f"[Warning] Image display failed at frame {frame_index}: {e}")

    def close(self) -> None:
        if self.cv2 is None:
            return
        try:
            self.cv2.destroyAllWindows()
        except Exception:
            pass
        self.enabled = False


def _validate_speed(speed: float) -> None:
    if speed <= 0:
        raise ValueError(f"speed 必须大于 0，当前为 {speed}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect historical HDF5 records without moving the robot"
    )
    parser.add_argument("--file", "-f", type=Path)
    args = parser.parse_args()
    path = args.file or Path(find_latest_log(_default_hdf5_dir()))
    trajectory = load_trajectory(path)
    for key, value in trajectory.items():
        print(key, getattr(value, "shape", value))
    print("Use the replay workspace for LeRobot image/audio/state playback.")


if __name__ == "__main__":
    main()
