#!/usr/bin/env python3
"""
使用手机控制 Franka 机械臂

支持 iOS (ARKit + HEBI Mobile I/O) 和 Android (WebXR)
基于 lerobot Phone teleoperator，完全遵循 robotic_arm_controller._teleop_control 的控制逻辑
"""

import argparse
import os
import time
import threading
import h5py
import numpy as np
from datetime import datetime
from typing import Optional

from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from robotic_arm_controller import (
    RoboticArmControler,
    _LatestFrameBuffer,
    _camera_capture_worker,
)
from realsense_connector import RealSenseConnector


def main():
    parser = argparse.ArgumentParser(description="用手机控制 Franka 机械臂")
    parser.add_argument(
        "--platform",
        type=str,
        choices=["ios", "android"],
        default="android",
        help="手机平台: ios (需要 HEBI Mobile I/O App) 或 android (使用 WebXR)",
    )
    parser.add_argument(
        "--control-frequency",
        type=float,
        default=10.0,
        help="控制频率 (Hz), 推荐: 10-50",
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="禁用数据记录（默认启用）",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="数据采集语言指令（logging 启用时必需）",
    )
    parser.add_argument(
        "--camera-timeout-ms",
        type=int,
        default=1000,
        help="相机采集超时（毫秒）",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=3600.0,
        help="最大记录时长（秒）",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Franka 机械臂手机控制")
    print("=" * 70)
    print(f"平台: {args.platform.upper()}")
    print(f"控制频率: {args.control_frequency} Hz")
    enable_logging = not args.no_logging
    if enable_logging:
        print(f"数据记录: 启用（最大 {args.max_duration} 秒）")
        print(f"语言指令: {args.instruction}")
    else:
        print(f"数据记录: 禁用")
    print("=" * 70)
    print()

    # 验证参数
    if enable_logging and not args.instruction.strip():
        raise ValueError("启用 logging 时必须提供 --instruction 参数")

    # 初始化 Phone 配置
    phone_os = PhoneOS.IOS if args.platform == "ios" else PhoneOS.ANDROID
    phone_config = PhoneConfig(phone_os=phone_os)

    # 初始化 Phone 设备
    print("正在连接手机...")
    if args.platform == "android":
        print("提示：程序启动后会打印本地 URL，请在手机浏览器中打开该 URL")
    phone = Phone(phone_config)
    phone.connect()  # 这会自动触发校准流程
    print("手机已连接并校准完成！\n")

    # 初始化机械臂
    print("正在初始化 Franka 机械臂...")
    arm = RoboticArmControler()

    # 初始化相机（如果需要记录）
    camera = None
    if enable_logging:
        print("正在初始化 RealSense 相机...")
        camera = RealSenseConnector()
        camera.connect()

    try:
        # 移动到起始位置
        print("正在移动到起始位置...")
        arm.move_to_start()

        # 打开夹爪
        print("正在打开夹爪...")
        arm.gripper_open()

        print("\n初始化完成!")
        print("\n控制说明:")
        if args.platform == "ios":
            print("  - 按住 B1: 启用控制")
            print("  - 释放 B1: 停止控制")
            print("  - A3 滑块: 控制夹爪速度（连续控制）")
            print("  - 按 Ctrl+C: 退出并保存数据")
        else:  # android
            print("  - 按住 Move 按钮: 启用控制")
            print("  - 释放 Move 按钮: 停止控制")
            print("  - reservedButtonA: 打开夹爪")
            print("  - reservedButtonB: 关闭夹爪")
            print("  - 按 Ctrl+C: 退出并保存数据")

        print("\n提示：")
        print("  - 首次按下启用按钮时会重新校准零点，避免累积漂移")
        print("  - 掐头去尾模式：检测到第一个动作开始记录，Ctrl+C 结束")
        print("=" * 70)
        print()

        # 运行控制循环
        _phone_control(
            arm=arm,
            phone=phone,
            camera=camera,
            control_frequency=args.control_frequency,
            enable_logging=enable_logging,
            max_logging_duration=args.max_duration,
            camera_timeout_ms=args.camera_timeout_ms,
            language_instruction=args.instruction,
            platform=args.platform,
        )

    except KeyboardInterrupt:
        print("\n\n检测到 Ctrl+C，正在退出...")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n正在清理资源...")
        phone.disconnect()
        if camera is not None:
            camera.close()
        arm.cleanup()
        print("清理完成！")


def _phone_control(
    arm: RoboticArmControler,
    phone: Phone,
    camera: Optional[RealSenseConnector],
    control_frequency: float,
    enable_logging: bool,
    max_logging_duration: float,
    camera_timeout_ms: int,
    language_instruction: str,
    platform: str,
) -> None:
    """
    手机控制实现（完全遵循 robotic_arm_controller._teleop_control 的逻辑）

    核心特性：
    1. ✅ 零点重置：当 enabled 从 False 变为 True 时，重新捕获手机基准姿态
    2. ✅ 夹爪漂移补偿：夹爪动作后重新同步位姿，并记录漂移
    3. ✅ 掐头去尾记录：检测到第一个动作开始记录，Ctrl+C 结束
    4. ✅ Reflex 错误处理：碰撞时删除损坏数据
    5. ✅ 相机同步采集：独立线程采集图像，跳过无效帧
    """
    from panda_py import controllers
    from transforms3d.quaternions import qmult, qinverse
    from transforms3d.euler import quat2euler
    from lerobot.utils.rotation import Rotation

    running = [True]  # 使用列表以便在回调中修改
    manual_stop = [False]  # 用于检测用户是否手动停止

    print("手机控制已启动！")
    print("控制说明：")
    if platform == "ios":
        print("  通过 HEBI Mobile I/O App 控制")
        print("  按住 B1 启用控制，A3 滑块控制夹爪")
    else:
        print("  通过 WebXR 控制")
        print("  按住 Move 启用控制，reservedButtonA/B 控制夹爪")
    print("  按 Ctrl+C 停止记录并退出")
    print("\n[Recording] 掐头去尾模式：检测到第一个动作开始记录，Ctrl+C 结束\n")

    # 使用实时阻抗控制
    ctrl = controllers.CartesianImpedance(filter_coeff=1.0)

    log_file = None
    h5_file = None
    image_ds = None
    action_ds = None
    instr_ds = None

    camera_stop = None
    camera_thread = None
    camera_buf = None

    if enable_logging and hasattr(arm, "log_dir"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(arm.log_dir, f"phone_control_{timestamp}.h5")
        h5_file = h5py.File(log_file, "w")
        arm._last_log_filename = log_file

        h5_file.attrs["created_at"] = timestamp
        h5_file.attrs["control_mode"] = f"phone_{platform}"
        h5_file.attrs["control_frequency_hz"] = float(control_frequency)
        h5_file.attrs["action_type"] = "EEF_POS"
        h5_file.attrs["image_center_crop_scale"] = 0.9
        h5_file.attrs["image_size_hw"] = 224

        obs_group = h5_file.require_group("observation")
        task_group = h5_file.require_group("task")
        image_ds = obs_group.create_dataset(
            "image_primary",
            shape=(0, 224, 224, 3),
            maxshape=(None, 224, 224, 3),
            chunks=(1, 224, 224, 3),
            dtype=np.uint8,
            compression="lzf",
        )
        action_ds = h5_file.create_dataset(
            "action",
            shape=(0, 7),
            maxshape=(None, 7),
            chunks=(256, 7),
            dtype=np.float32,
            compression="lzf",
        )
        instr_ds = task_group.create_dataset(
            "language_instruction",
            shape=(0,),
            maxshape=(None,),
            chunks=(256,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

        if camera is None:
            raise ValueError(
                "Phone control logging requires camera=RealSenseConnector()."
            )

        camera_buf = _LatestFrameBuffer()
        camera_stop = threading.Event()
        camera_thread = threading.Thread(
            target=_camera_capture_worker,
            kwargs={
                "camera": camera,
                "buf": camera_buf,
                "stop_event": camera_stop,
                "timeout_ms": int(camera_timeout_ms),
                "crop_scale": 0.9,
                "out_hw": 224,
            },
            daemon=True,
        )
        camera_thread.start()

        # ✅ 等待相机线程采集到第一帧
        print("[Camera] Waiting for first frame...")
        max_wait_time = 5.0
        wait_start = time.time()
        while camera_buf.get_latest() is None:
            if time.time() - wait_start > max_wait_time:
                raise RuntimeError(
                    "Camera timeout: Failed to get first frame after 5 seconds. "
                    "Please check camera connection."
                )
            time.sleep(0.01)
        print("[Camera] First frame acquired, ready to record!")

    img_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    instr_buf: list[str] = []
    flush_every = 20

    # ✅ 记录控制：只有检测到移动才开始记录
    recording_started = False
    recording_stopped = False

    # ✅ 在启动控制器前读取初始位姿
    target_position = arm.panda.get_position().astype(np.float64)
    target_orientation = arm.panda.get_orientation().astype(
        np.float64
    )  # 四元数 [w, x, y, z]
    gripper_state = 1.0
    last_gripper_cmd = 1.0  # 记录上一次的夹爪命令

    # ✅ Phone -> EE 映射与安全限制（替代 lerobot pipeline 的 Map/EEBounds/EEReference 步骤）
    # 速度缩放：用于调节手机位移/旋转映射到机械臂的幅度
    pos_scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    rot_scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    # 工作空间限制：以初始位姿为中心设置安全边界（可按需调大/调小）
    workspace_half_extent = np.array([0.3, 0.3, 0.3], dtype=np.float64)
    workspace_min = target_position - workspace_half_extent
    workspace_max = target_position + workspace_half_extent
    # 单步最大移动距离（防止瞬移）
    max_ee_step_m = 0.05

    # ✅ 零点偏移：记录手机姿态的基准点（当 enabled 从 False 变为 True 时记录）
    phone_base_pos = None  # 3D 位置，作为手机的"零点"
    phone_base_rot_inv = None  # Rotation 对象的逆，作为旋转基准
    last_enabled = False  # 记录上一次的 enabled 状态

    # 保存初始位姿到 HDF5（用于回放验证）
    if h5_file is not None:
        h5_file.attrs["initial_position"] = target_position.tolist()
        h5_file.attrs["initial_orientation"] = target_orientation.tolist()
        print(f"[Data] Initial position: {target_position}")
        print(f"[Data] Initial orientation: {target_orientation}")

    identity_rot = Rotation.from_rotvec(np.zeros(3))
    arm.panda.start_controller(ctrl)

    # 标志：是否发生了 reflex 错误（用于决定是否保存数据）
    reflex_error_occurred = False

    try:
        with arm.panda.create_context(frequency=control_frequency) as ctx:
            while ctx.ok() and running[0]:
                # 从 Phone 获取最新动作
                action = phone.get_action()

                if not action:
                    # 没有接收到有效动作，继续等待
                    time.sleep(0.001)
                    continue

                # 解析 Phone 动作
                phone_pos = action.get("phone.pos", np.zeros(3))  # 校准后的相对位置 (m)
                phone_rot = action.get("phone.rot", identity_rot)  # 校准后的相对旋转
                raw_inputs = action.get("phone.raw_inputs", {})
                enabled = action.get("phone.enabled", False)  # 是否正在控制

                # ✅ 零点重置：当 enabled 从 False 变为 True 时，记录当前手机姿态作为基准
                # 这与 _teleop_control 中的 "is_moving and not last_is_moving" 逻辑一致
                if enabled and not last_enabled:
                    phone_base_pos = phone_pos.copy()
                    phone_base_rot_inv = phone_rot.inv()
                    print(f"\n[Phone] 检测到 enabled=True，已重置手机零点基准")
                    print(
                        f"  基准位置: [{phone_base_pos[0]:.3f}, {phone_base_pos[1]:.3f}, {phone_base_pos[2]:.3f}]"
                    )

                last_enabled = enabled

                # ✅ 计算相对于基准点的增量（只有在 enabled=True 且已设置基准点时才计算）
                if (
                    enabled
                    and phone_base_pos is not None
                    and phone_base_rot_inv is not None
                ):
                    # 计算手机的相对位移: delta_pos = rot_base_inv * (pos - pos_base)
                    delta_pos = phone_base_rot_inv.apply(phone_pos - phone_base_pos)
                    delta_pos = delta_pos * pos_scale

                    # 计算手机的相对旋转: delta_rot = rot_base_inv * rot
                    delta_rot = phone_base_rot_inv * phone_rot
                    delta_rot = Rotation.from_rotvec(delta_rot.as_rotvec() * rot_scale)

                    # 将手机的相对变换应用到机械臂的当前位姿
                    target_position_new = target_position + delta_pos

                    # 转换 Rotation 对象为四元数并应用
                    delta_quat = delta_rot.as_quat()  # scipy Rotation: [x, y, z, w]
                    delta_quat_wxyz = np.array(
                        [delta_quat[3], delta_quat[0], delta_quat[1], delta_quat[2]],
                        dtype=np.float64,
                    )  # [w, x, y, z]
                    target_orientation_new = qmult(target_orientation, delta_quat_wxyz)
                else:
                    # 如果没有启用或者没有基准点，保持当前位姿
                    target_position_new = target_position
                    target_orientation_new = target_orientation

                # ✅ 安全限制：限制单步位移并夹紧到工作空间
                step = target_position_new - target_position
                step_norm = np.linalg.norm(step)
                if step_norm > max_ee_step_m:
                    target_position_new = target_position + step / step_norm * max_ee_step_m
                target_position_new = np.minimum(
                    np.maximum(target_position_new, workspace_min), workspace_max
                )

                # 计算增量动作（用于记录）
                delta = np.zeros(6, dtype=np.float64)
                delta[:3] = target_position_new - target_position

                # 计算旋转增量
                delta_quat_for_record = qmult(
                    target_orientation_new, qinverse(target_orientation)
                )
                delta[3:] = quat2euler(delta_quat_for_record, axes="sxyz")

                # 处理夹爪命令（根据平台不同）
                gripper_drift_delta = np.zeros(6, dtype=np.float64)
                gripper_cmd = last_gripper_cmd  # 默认保持上一次的命令

                if platform == "ios":
                    # iOS: A3 滑块控制夹爪（-1 到 1）
                    gripper_analog = float(raw_inputs.get("a3", 0.0))
                    # 映射到 0-1: -1=关闭(0), 1=打开(1)
                    gripper_cmd = (gripper_analog + 1.0) / 2.0
                else:  # android
                    # Android: reservedButtonA/B 控制夹爪
                    if raw_inputs.get("reservedButtonA", False):  # 打开
                        gripper_cmd = 1.0
                    elif raw_inputs.get("reservedButtonB", False):  # 关闭
                        gripper_cmd = 0.0

                if gripper_cmd != last_gripper_cmd:
                    # 夹爪状态改变
                    pos_before = arm.panda.get_position().astype(np.float64)
                    ori_before = arm.panda.get_orientation().astype(np.float64)

                    arm.panda.stop_controller()
                    if gripper_cmd > 0.5:  # 打开
                        arm.gripper_open()
                        gripper_state = 1.0
                    else:  # 关闭
                        arm.gripper_close()
                        gripper_state = 0.0

                    # 记录夹爪动作后的位置
                    target_position = arm.panda.get_position().astype(np.float64)
                    target_orientation = arm.panda.get_orientation().astype(np.float64)

                    # 计算位置偏移（用于记录）
                    gripper_drift_delta[:3] = target_position - pos_before
                    delta_quat_gripper = qmult(target_orientation, qinverse(ori_before))
                    gripper_drift_delta[3:] = quat2euler(
                        delta_quat_gripper, axes="sxyz"
                    )

                    arm.panda.start_controller(ctrl)
                    last_gripper_cmd = gripper_cmd

                # 如果夹爪动作导致了位置偏移，将偏移量加到 delta 中
                if np.any(gripper_drift_delta != 0):
                    delta = delta + gripper_drift_delta

                # ✅ 先检查相机图像（如果需要记录）
                skip_frame = False
                if (
                    h5_file is not None
                    and camera_buf is not None
                    and image_ds is not None
                    and action_ds is not None
                    and instr_ds is not None
                ):
                    rgb224 = camera_buf.get_latest()
                    if rgb224 is None:
                        print(
                            "[Warning] Camera frame is None, skipping this frame",
                            end="\r",
                        )
                        skip_frame = True
                    elif rgb224.shape != (224, 224, 3) or rgb224.dtype != np.uint8:
                        print(
                            f"[Warning] Invalid frame shape/dtype: {rgb224.shape}/{rgb224.dtype}",
                            end="\r",
                        )
                        skip_frame = True

                if skip_frame:
                    continue

                # 应用增量动作 - 只有当 enabled 为 True 时才移动
                if enabled:
                    # 更新目标位姿
                    target_position = target_position_new
                    target_orientation = target_orientation_new

                    # 设置控制目标
                    ctrl.set_control(target_position, target_orientation)

                # 记录数据（此时已确保有图像）
                # ✅ 掐头去尾：检测到移动开始记录，Ctrl+C 结束
                if (
                    h5_file is not None
                    and camera_buf is not None
                    and image_ds is not None
                    and action_ds is not None
                    and instr_ds is not None
                    and not recording_stopped
                ):
                    # 检查是否有动作
                    has_action = np.any(delta != 0) or enabled

                    # 第一次有动作时开始记录（掐头）
                    if has_action and not recording_started:
                        recording_started = True
                        print("[Recording] 检测到第一个动作，开始记录数据...")

                    # 开始记录后，保留所有帧
                    if recording_started:
                        action7 = np.concatenate(
                            [
                                delta.astype(np.float32, copy=False),
                                [np.float32(gripper_state)],
                            ]
                        )
                        # rgb224 已经过检查，确保不为 None
                        assert rgb224 is not None
                        img_buf.append(rgb224)
                        act_buf.append(action7.astype(np.float32, copy=False))
                        instr_buf.append(str(language_instruction))

                    if len(img_buf) >= flush_every:
                        n0 = int(image_ds.shape[0])
                        batch_n = len(img_buf)
                        image_ds.resize((n0 + batch_n, 224, 224, 3))
                        action_ds.resize((n0 + batch_n, 7))
                        instr_ds.resize((n0 + batch_n,))
                        image_ds[n0 : n0 + batch_n] = np.stack(img_buf, axis=0)
                        action_ds[n0 : n0 + batch_n] = np.stack(act_buf, axis=0)
                        instr_ds[n0 : n0 + batch_n] = np.asarray(
                            instr_buf, dtype=object
                        )
                        img_buf.clear()
                        act_buf.clear()
                        instr_buf.clear()
                        print(f"[Recording] 已记录 {n0 + batch_n} 帧", end="\r")

    except RuntimeError as e:
        # 检查是否是 reflex 错误
        error_msg = str(e)
        if "cartesian_reflex" in error_msg or "motion aborted by reflex" in error_msg:
            print(f"\n[Error] Cartesian reflex triggered: {error_msg}")
            print("[Error] Data will NOT be saved due to reflex error")
            reflex_error_occurred = True
        raise
    except KeyboardInterrupt:
        print("\n[Recording] 检测到 Ctrl+C，停止记录...")
        manual_stop[0] = True
        running[0] = False
    finally:
        arm.panda.stop_controller()
        if camera_stop is not None:
            camera_stop.set()
        if camera_thread is not None:
            camera_thread.join(timeout=2.0)

        # 如果发生 reflex 错误，删除文件
        if reflex_error_occurred:
            if h5_file is not None:
                try:
                    h5_file.close()
                except Exception:
                    pass
            if log_file is not None:
                try:
                    os.remove(log_file)
                    print(f"[Cleanup] Deleted corrupted data file: {log_file}")
                except Exception as e:
                    print(f"[Cleanup] Failed to delete file: {e}")
        # 没有 reflex 错误时正常保存数据
        elif (
            h5_file is not None
            and image_ds is not None
            and action_ds is not None
            and instr_ds is not None
        ):
            if len(img_buf) > 0:
                n0 = int(image_ds.shape[0])
                batch_n = len(img_buf)
                image_ds.resize((n0 + batch_n, 224, 224, 3))
                action_ds.resize((n0 + batch_n, 7))
                instr_ds.resize((n0 + batch_n,))
                image_ds[n0 : n0 + batch_n] = np.stack(img_buf, axis=0)
                action_ds[n0 : n0 + batch_n] = np.stack(act_buf, axis=0)
                instr_ds[n0 : n0 + batch_n] = np.asarray(instr_buf, dtype=object)
            total_frames = int(image_ds.shape[0])
            try:
                h5_file.close()
            except Exception:
                pass
            if log_file is not None:
                print(f"\n[Recording] 记录完成！总计 {total_frames} 帧")
                print(f"数据已保存到: {log_file}")

    print("\n手机控制已退出")


if __name__ == "__main__":
    main()
