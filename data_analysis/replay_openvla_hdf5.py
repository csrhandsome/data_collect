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
import sys
import time
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
from spatialmath import SE3

from control.robotic_arm_controller import RoboticArmControler


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
                trajectory_data["initial_orientation"] = list(
                    f.attrs["initial_orientation"]
                )

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
            live_img_bgr = self.cv2.cvtColor(
                self.camera.img.copy(), self.cv2.COLOR_RGB2BGR
            )
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


def _confirm_initial_pose(arm: RoboticArmControler, trajectory: dict) -> bool:
    if "initial_position" not in trajectory or "initial_orientation" not in trajectory:
        return True

    recorded_pos = np.array(trajectory["initial_position"], dtype=np.float64)
    recorded_ori = np.array(trajectory["initial_orientation"], dtype=np.float64)
    current_pos = arm.panda.get_position().astype(np.float64)
    current_ori = arm.panda.get_orientation().astype(np.float64)

    pos_diff = np.linalg.norm(recorded_pos - current_pos)
    ori_diff = np.linalg.norm(recorded_ori - current_ori)

    print("\n初始位姿验证:")
    print(f"  位置差异: {pos_diff:.4f} m")
    print(f"  姿态差异: {ori_diff:.4f} (四元数)")

    if pos_diff > 0.05:
        print(f"  ⚠️ 警告: 初始位置差异较大 ({pos_diff * 1000:.1f} mm)")
        print(f"  记录位置: {recorded_pos}")
        print(f"  当前位置: {current_pos}")
        response = input("  继续回放可能导致轨迹偏差，是否继续? (y/n): ").strip().lower()
        if response != "y":
            print("  取消回放")
            return False
    elif pos_diff > 0.01:
        print(
            f"  ⚠️ 提示: 初始位置有微小差异 ({pos_diff * 1000:.1f} mm)，回放可能略有偏差"
        )
    return True


def _prompt_next_loop(loop_idx: int, loop_count: int) -> bool:
    if loop_idx >= loop_count - 1:
        return True
    try:
        response = input("\n继续下一次播放？(Y/n): ").strip().lower()
        if response == "n":
            print("取消剩余播放")
            return False
    except KeyboardInterrupt:
        print("\n取消剩余播放")
        return False
    return True


def _print_action_replay_diagnostics(
    arm: RoboticArmControler,
    *,
    played_frames: int,
    total_frames: int,
    ctx_ok: bool,
) -> None:
    print(
        f"\n⚠️ 警告: 播放未完成! 播放了 {played_frames}/{total_frames} 帧 ({played_frames / total_frames * 100:.1f}%)"
    )
    print(f"  ctx.ok() = {ctx_ok}")

    try:
        robot_state = arm.panda.get_robot().read_once()
        print("  机器人状态检查:")

        if hasattr(robot_state, "cartesian_collision"):
            collision = any(robot_state.cartesian_collision)
            print(f"    碰撞检测: {collision}")

        if hasattr(robot_state, "q"):
            q = robot_state.q
            print(f"    当前关节位置: [{', '.join(f'{x:.3f}' for x in q[:3])}...]")

        pose = arm.panda.get_pose()
        if pose is not None:
            transform = SE3(pose, check=False)
            pos = transform.t
            print(f"    当前末端位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    except Exception as e:
        print(f"  无法读取详细状态: {e}")

    print("\n可能原因:")
    print("  1. 碰撞检测触发（机器人检测到碰撞）")
    print("  2. 工作空间边界（超出机器人可达范围）")
    print("  3. 关节限位（某个关节超出限制）")
    print("  4. 控制器超时（动作执行时间过长）")
    print("  5. 初始位置不匹配（回放起点与采集起点不同）")


def _replay_action_trajectory(
    arm: RoboticArmControler,
    trajectory: dict,
    *,
    speed: float,
    loop: int,
    start_index: int,
    end_index: int | None,
    step_callback: Callable[[dict], None] | None,
) -> None:
    from panda_py import controllers
    from transforms3d.euler import euler2mat
    from transforms3d.quaternions import mat2quat, qmult

    actions = np.asarray(trajectory["action"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected action shape (T, 7), got {actions.shape}")

    stop = len(actions) if end_index is None else end_index
    actions = actions[start_index:stop]
    if len(actions) == 0:
        raise ValueError("动作序列为空，无法回放")

    hz = float(trajectory.get("control_frequency_hz", 100.0))

    if not _confirm_initial_pose(arm, trajectory):
        return

    print(f"\n{'=' * 60}")
    print("动作序列重放 (EEF_POS delta + gripper)")
    print(f"{'=' * 60}")
    print(f"轨迹长度: {len(actions)} 个样本")
    print(f"播放速度: {speed}x")
    print(f"循环次数: {loop}")
    print(f"采样频率: {hz:.1f} Hz")
    print(f"{'=' * 60}\n")

    for loop_idx in range(loop):
        if loop > 1:
            print(f"\n--- 第 {loop_idx + 1}/{loop} 次播放 ---")

        ctrl = controllers.CartesianImpedance(filter_coeff=1.0)
        target_position = arm.panda.get_position().astype(np.float64)
        target_orientation = arm.panda.get_orientation().astype(np.float64)
        last_gripper_open: bool | None = None

        arm.panda.start_controller(ctrl)
        try:
            freq = max(1.0, hz * float(speed))
            max_runtime = len(actions) / hz / float(speed) if hz > 0 else None
            ctx_kwargs = {"frequency": freq}
            if max_runtime is not None:
                ctx_kwargs["max_runtime"] = max_runtime

            print(f"[Replay] 控制参数: 频率={freq:.1f}Hz, 最大时长={max_runtime:.1f}秒")

            i = 0
            last_error_check = 0
            ctx_ok = False
            with arm.panda.create_context(**ctx_kwargs) as ctx:
                while ctx.ok() and i < len(actions):
                    if i - last_error_check >= 100:
                        try:
                            robot_state = arm.panda.get_robot().read_once()
                            if hasattr(robot_state, "robot_mode"):
                                pass
                            last_error_check = i
                        except Exception as e:
                            print(f"\n[Warning] 无法读取机器人状态: {e}")

                    action = actions[i].astype(np.float64, copy=False)
                    delta = action[:6]
                    gripper_score = float(action[6])
                    gripper_open = gripper_score >= 0.5

                    if last_gripper_open is None or gripper_open != last_gripper_open:
                        arm.panda.stop_controller()
                        if gripper_open:
                            arm.gripper_open()
                        else:
                            arm.gripper_close()
                        target_position = arm.panda.get_position().astype(np.float64)
                        target_orientation = arm.panda.get_orientation().astype(np.float64)
                        arm.panda.start_controller(ctrl)
                        last_gripper_open = gripper_open

                    if np.any(delta != 0):
                        target_position = target_position + delta[:3]
                        if np.any(delta[3:] != 0):
                            delta_quat = mat2quat(
                                euler2mat(delta[3], delta[4], delta[5])
                            )
                            target_orientation = qmult(target_orientation, delta_quat)
                        ctrl.set_control(target_position, target_orientation)

                    if step_callback is not None:
                        try:
                            step_callback(
                                {
                                    "frame_index": i,
                                    "frame_count": len(actions),
                                    "loop_index": loop_idx,
                                    "loop_count": loop,
                                    "gripper_open": gripper_open,
                                }
                            )
                        except Exception as e:
                            if i == 0 or i % 100 == 0:
                                print(f"[Warning] Step callback failed at frame {i}: {e}")

                    i += 1
                    if i % 100 == 0:
                        progress = (i / len(actions)) * 100
                        print(f"进度: {progress:.1f}% ({i}/{len(actions)})", end="\r")

                ctx_ok = ctx.ok()

            if i < len(actions):
                _print_action_replay_diagnostics(
                    arm,
                    played_frames=i,
                    total_frames=len(actions),
                    ctx_ok=ctx_ok,
                )
            else:
                print(f"\n✓ 播放完成 ({i}/{len(actions)} 样本)")
        except KeyboardInterrupt:
            print("\n中断播放")
            break
        finally:
            arm.panda.stop_controller()

        if not _prompt_next_loop(loop_idx, loop):
            break

    print("\n✓ 重放完成")


def _replay_joint_trajectory(
    arm: RoboticArmControler,
    trajectory: dict,
    *,
    speed: float,
    loop: int,
    start_index: int,
    end_index: int | None,
) -> None:
    from panda_py import controllers

    q = np.asarray(trajectory["q"])
    dq = np.asarray(trajectory["dq"])
    stop = len(q) if end_index is None else end_index
    q = q[start_index:stop]
    dq = dq[start_index:stop]

    if len(q) == 0:
        raise ValueError("轨迹为空，无法回放")

    print(f"\n{'=' * 60}")
    print("轨迹重放")
    print(f"{'=' * 60}")
    print(f"轨迹长度: {len(q)} 个样本")
    print(f"播放速度: {speed}x")
    print(f"循环次数: {loop}")
    print(f"{'=' * 60}\n")

    print("正在移动到轨迹起始位置...")
    arm.panda.move_to_joint_position(q[0])
    print("已到达起始位置\n")

    input("按 Enter 键开始重放轨迹...")

    for loop_idx in range(loop):
        if loop > 1:
            print(f"\n--- 第 {loop_idx + 1}/{loop} 次播放 ---")

        base_frequency = 1000.0
        control_frequency = base_frequency * speed
        max_runtime = len(q) / base_frequency / speed

        ctrl = controllers.JointPosition()
        arm.panda.start_controller(ctrl)

        try:
            i = 0
            with arm.panda.create_context(
                frequency=control_frequency,
                max_runtime=max_runtime,
            ) as ctx:
                while ctx.ok() and i < len(q):
                    ctrl.set_control(q[i], dq[i])
                    i += 1

                    if i % 100 == 0:
                        progress = (i / len(q)) * 100
                        print(f"进度: {progress:.1f}% ({i}/{len(q)})", end="\r")

            print(f"\n✓ 播放完成 ({i}/{len(q)} 样本)")
        except KeyboardInterrupt:
            print("\n中断播放")
            break
        except Exception as e:
            print(f"\n播放出错: {e}")
            raise
        finally:
            arm.panda.stop_controller()

        if not _prompt_next_loop(loop_idx, loop):
            break

    print("\n✓ 重放完成")


def replay_trajectory(
    arm: RoboticArmControler,
    trajectory: dict,
    speed: float = 1.0,
    loop: int = 1,
    start_index: int = 0,
    end_index: int | None = None,
    camera=None,
    show_images: bool = True,
    show_live_camera: bool = False,
) -> None:
    _validate_speed(speed)

    display = _ReplayDisplayManager(
        recorded_images=_slice_recorded_images(trajectory, start_index, end_index),
        camera=camera,
        show_images=show_images,
        show_live_camera=show_live_camera,
    )
    display.setup()

    try:
        if "action" in trajectory:
            _replay_action_trajectory(
                arm,
                trajectory,
                speed=speed,
                loop=loop,
                start_index=start_index,
                end_index=end_index,
                step_callback=display.render_step if display.enabled else None,
            )
        else:
            _replay_joint_trajectory(
                arm,
                trajectory,
                speed=speed,
                loop=loop,
                start_index=start_index,
                end_index=end_index,
            )
    finally:
        display.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="重放机械臂轨迹")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="HDF5轨迹文件路径（不指定则自动查找最新的）",
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=float,
        default=1.0,
        help="播放速度倍率 (0.1-2.0)，默认1.0",
    )
    parser.add_argument(
        "--loop",
        "-l",
        type=int,
        default=1,
        help="循环播放次数，默认1次",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="起始样本索引，默认0",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="结束样本索引，默认到末尾",
    )
    parser.add_argument(
        "--hide-images",
        action="store_true",
        help="不显示记录时保存的图像",
    )
    parser.add_argument(
        "--show-live-camera",
        action="store_true",
        help="显示实时相机对比",
    )
    parser.add_argument(
        "--no-init",
        action="store_true",
        help="跳过初始化（假设机械臂已就绪）",
    )

    args = parser.parse_args()

    if not (0.1 <= args.speed <= 2.0):
        print("警告: 播放速度超出推荐范围 [0.1, 2.0]，可能导致运动不稳定")

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"错误: 文件不存在: {file_path}")
            sys.exit(1)
    else:
        try:
            file_path = Path(find_latest_log(_default_hdf5_dir()))
            print(f"自动选择最新的日志文件: {file_path}")
        except FileNotFoundError as e:
            print(f"错误: {e}")
            sys.exit(1)

    try:
        trajectory = load_trajectory(file_path)
    except Exception as e:
        print(f"错误: 加载轨迹失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    show_images = not args.hide_images
    show_live_camera = args.show_live_camera

    arm = None
    camera = None

    try:
        print("\n正在初始化机械臂...")
        arm = RoboticArmControler()

        if show_live_camera:
            print("正在初始化相机...")
            from control.camera_connector import RealSenseConnector

            camera = RealSenseConnector()
            print("✓ 相机初始化成功")

        if not args.no_init:
            print("正在移动到起始位置...")
            arm.move_to_start()

        replay_trajectory(
            arm=arm,
            trajectory=trajectory,
            speed=args.speed,
            loop=args.loop,
            start_index=args.start,
            end_index=args.end,
            camera=camera,
            show_images=show_images,
            show_live_camera=show_live_camera,
        )

        print("\n✓ 所有播放完成")
    except KeyboardInterrupt:
        print("\n\n检测到中断信号，正在退出...")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n正在清理资源...")
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
        if arm is not None:
            arm.cleanup()
        print("清理完成！")


if __name__ == "__main__":
    main()
