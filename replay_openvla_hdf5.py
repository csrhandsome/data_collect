#!/usr/bin/env python3
"""
重放（Replay）之前记录的机械臂轨迹

使用说明:
1. 自动查找最新的记录: python replay_demo.py
2. 指定HDF5文件: python replay_demo.py --file path/to/file.h5
3. 调整播放速度: python replay_demo.py --speed 0.5  # 0.5倍速
4. 循环播放: python replay_demo.py --loop 3  # 播放3次

数据来源:
- manual_control 记录的 HDF5 文件
- 位于 data/hdf5/ 目录下
"""

import argparse
import os
import sys
from pathlib import Path
import h5py
import numpy as np
from control.robotic_arm_controller import RoboticArmControler


def find_latest_log(log_dir: str) -> str:
    """查找最新的日志文件"""
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"日志目录不存在: {log_dir}")

    h5_files = list(log_path.glob("gamepad_control_*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"在 {log_dir} 中未找到任何 gamepad_control_*.h5 文件")

    # 按修改时间排序，返回最新的
    latest_file = max(h5_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


def load_trajectory(file_path: str):
    """从HDF5文件加载轨迹数据"""
    print(f"正在加载轨迹: {file_path}")

    with h5py.File(file_path, "r") as f:
        print(f"可用数据集: {list(f.keys())}")

        if "action" in f:
            actions = np.array(f["action"], dtype=np.float32)
            hz = float(f.attrs.get("control_frequency_hz", 100.0))
            print(f"动作序列长度: {len(actions)} 个样本, 采样频率: {hz:.1f} Hz")

            trajectory_data = {"action": actions, "control_frequency_hz": hz}

            # 读取图像数据（如果有）
            if "observation" in f and "image_primary" in f["observation"]:
                images = np.array(f["observation/image_primary"], dtype=np.uint8)
                trajectory_data["images"] = images
                print(f"✓ 加载了 {len(images)} 帧图像数据")
            else:
                print(f"⚠️ 该文件没有图像数据")

            # 读取初始位姿（如果有）
            if "initial_position" in f.attrs:
                trajectory_data["initial_position"] = list(f.attrs["initial_position"])
            if "initial_orientation" in f.attrs:
                trajectory_data["initial_orientation"] = list(
                    f.attrs["initial_orientation"]
                )

            return trajectory_data

        # 兼容旧格式（q/dq）
        q = np.array(f["q"])
        dq = np.array(f["dq"]) if "dq" in f else np.zeros_like(q)
        time = np.array(f["time"]) if "time" in f else None
        print(f"轨迹长度: {len(q)} 个样本")
        if time is not None and len(time) > 1:
            duration = float(time[-1] - time[0])
            if duration > 0:
                print(f"轨迹时长: {duration:.2f} 秒")
                print(f"采样频率: {len(q) / duration:.1f} Hz")
        return {"q": q, "dq": dq, "time": time}


def replay_trajectory(
    arm: RoboticArmControler,
    trajectory: dict,
    speed: float = 1.0,
    loop: int = 1,
    start_index: int = 0,
    end_index: int = None,
    camera=None,
    show_images: bool = True,
    show_live_camera: bool = False,
):
    arm.replay_trajectory(
        trajectory_data=trajectory,
        speed=speed,
        loop=loop,
        start_index=start_index,
        end_index=end_index,
        camera=camera,
        show_images=show_images,
        show_live_camera=show_live_camera,
    )


def main():
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
        "--no-init",
        action="store_true",
        help="跳过初始化（假设机械臂已就绪）",
    )

    args = parser.parse_args()

    # 验证速度参数
    if not (0.1 <= args.speed <= 2.0):
        print("警告: 播放速度超出推荐范围 [0.1, 2.0]，可能导致运动不稳定")

    # 查找或验证文件
    if args.file:
        file_path = args.file
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            sys.exit(1)
    else:
        # 自动查找最新的日志文件
        log_dir = os.path.join(os.path.dirname(__file__), "data/hdf5")
        try:
            file_path = find_latest_log(log_dir)
            print(f"自动选择最新的日志文件: {file_path}")
        except FileNotFoundError as e:
            print(f"错误: {e}")
            sys.exit(1)

    # 加载轨迹数据
    try:
        trajectory = load_trajectory(file_path)
    except Exception as e:
        print(f"错误: 加载轨迹失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 初始化机械臂
    print("\n正在初始化机械臂...")
    arm = RoboticArmControler()

    # 初始化相机（如果需要实时对比）
    camera = None
    print("正在初始化相机...")
    from realsense_connector import RealSenseConnector

    camera = RealSenseConnector()
    print("✓ 相机初始化成功")

    try:
        if not args.no_init:
            print("正在移动到起始位置...")
            arm.move_to_start()

        # 重放轨迹
        replay_trajectory(
            arm=arm,
            trajectory=trajectory,
            speed=args.speed,
            loop=args.loop,
            start_index=args.start,
            end_index=args.end,
            camera=camera,
            show_images=True,  # 显示采集的图像
            show_live_camera=False,  # 同时显示实时相机（对比）
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
        arm.cleanup()
        print("清理完成！")


if __name__ == "__main__":
    main()
