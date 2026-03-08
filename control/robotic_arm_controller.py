# 模块加载更快
from __future__ import annotations

import json
import os
import threading
from contextlib import suppress
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional, Sequence
import time
import h5py
import numpy as np
import panda_py
from spatialmath import SE3
from panda_py import libfranka

if TYPE_CHECKING:
    from control.camera_connector import RealSenseConnector


class _LatestFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._rgb: Optional[np.ndarray] = None

    def set(self, rgb: np.ndarray) -> None:
        with self._lock:
            self._seq += 1
            self._rgb = np.asarray(rgb, dtype=np.uint8)

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._rgb is None:
                return None
            return np.array(self._rgb, copy=True)


def _camera_capture_worker(
    *,
    camera: "RealSenseConnector",
    buf: _LatestFrameBuffer,
    stop_event: threading.Event,
    timeout_ms: int,
    crop_scale: float,
    out_hw: int,
    label: str | None = None,
) -> None:
    frame_count = 0
    error_count = 0
    name = label or "camera"
    while not stop_event.is_set():
        try:
            ok = camera.update(timeout=int(timeout_ms))
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"[Camera Thread:{name}] Update exception: {e}")
            ok = False
        if not ok:
            continue

        rgb = camera.img
        if rgb is None:
            continue
        try:
            rgb224 = _center_crop_and_resize_rgb_uint8(
                np.asarray(rgb, dtype=np.uint8),
                crop_scale=float(crop_scale),
                out_hw=int(out_hw),
            )
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"[Camera Thread:{name}] Crop/resize exception: {e}")
            continue
        buf.set(rgb224)
        frame_count += 1
        if frame_count == 1:
            print(f"[Camera Thread:{name}] First frame captured successfully!")
        elif frame_count % 100 == 0:
            print(f"[Camera Thread:{name}] {frame_count} frames captured")


def _center_crop_and_resize_rgb_uint8(
    rgb: np.ndarray, *, crop_scale: float = 0.9, out_hw: int = 224
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image of shape (H, W, 3), got {rgb.shape}")

    try:
        import tensorflow as tf

        # Match OpenVLAPolicy._crop_and_resize behavior (center crop by area scale, then resize to 224x224).
        image_tf = tf.convert_to_tensor(rgb)  # uint8 (H,W,3)
        orig_dtype = image_tf.dtype
        image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)  # [0,1]
        image_tf = tf.expand_dims(image_tf, axis=0)  # (1,H,W,3)

        batch_size = 1
        scale = tf.reshape(
            tf.clip_by_value(tf.sqrt(float(crop_scale)), 0, 1), shape=(batch_size,)
        )
        height_offsets = (1 - scale) / 2
        width_offsets = (1 - scale) / 2
        boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + scale,
                width_offsets + scale,
            ],
            axis=1,
        )

        image_tf = tf.image.crop_and_resize(
            image_tf, boxes, tf.range(batch_size), (int(out_hw), int(out_hw))
        )
        image_tf = image_tf[0]
        image_tf = tf.clip_by_value(image_tf, 0, 1)
        image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
        out = image_tf.numpy()
    except Exception:
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        frac = float(np.sqrt(float(crop_scale)))
        crop_h = max(1, min(h, int(round(h * frac))))
        crop_w = max(1, min(w, int(round(w * frac))))
        y0 = max(0, (h - crop_h) // 2)
        x0 = max(0, (w - crop_w) // 2)
        cropped = rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]

        from PIL import Image

        img = Image.fromarray(cropped, mode="RGB").resize(
            (int(out_hw), int(out_hw)), resample=Image.BILINEAR
        )
        out = np.asarray(img, dtype=np.uint8)

    if out.shape != (int(out_hw), int(out_hw), 3):
        raise RuntimeError(f"Unexpected resized RGB shape: {out.shape}")
    return out.astype(np.uint8, copy=False)


class RoboticArmControler:
    """利用panda_py这个无敌的好库来帮助我控制机械臂"""

    def __init__(
        self,
        hostname: str = "192.168.1.100",
        username: str = "BionicDL-Franka",
        password: str = "Design@2018",
        cartesian_speed_factor: float = 0.2,
        joint_speed_factor: float = 0.21,
        default_height: Optional[float] = 0.118046,
        enforce_default_height: bool = False,
        stiffness: Optional[Sequence[float]] = (600, 600, 600, 600, 250, 150, 50),
        damping: Optional[Sequence[float]] = None,
        auto_set_default_behavior: bool = True,
        auto_move_to_start: bool = True,
    ) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            hostname = config.get("hostname", hostname)
            username = config.get("username", username)
            password = config.get("password", password)

        # 启动 FCI 模式
        self.desk = panda_py.Desk(hostname, username, password)
        self.desk.unlock()
        self.desk.activate_fci()
        print("FCI 模式已激活")

        # 启动机械臂
        self.panda = panda_py.Panda(hostname)
        self.gripper = libfranka.Gripper(hostname)
        self.auto_set_default_behavior = auto_set_default_behavior
        self.auto_move_to_start = auto_move_to_start

        # 初始化机械臂参数
        self.T_0 = SE3(self.panda.get_pose(), check=False)
        self.joint_speed_factor = joint_speed_factor
        self.cart_speed_factor = cartesian_speed_factor
        self.stiffness = list(stiffness) if stiffness is not None else None
        self.damping = list(damping) if damping is not None else None
        self.default_height = default_height
        self.enforce_default_height = enforce_default_height

        # 设置碰撞检测阈值
        collision_torque = [80.0, 80.0, 80.0, 80.0, 48.0, 36.0, 24.0]
        collision_force = [90.0, 90.0, 90.0, 55.0, 55.0, 55.0]
        self.panda.get_robot().set_collision_behavior(
            collision_torque,
            collision_torque,
            collision_force,
            collision_force,
        )

        self.log_dir = os.path.join(os.path.dirname(__file__), "../../data/hdf5")
        os.makedirs(self.log_dir, exist_ok=True)

        self.robot_state: dict[str, np.ndarray] = {}
        self.time: list[float] = []
        self._logging_started_at: Optional[datetime] = None
        self._last_log_filename: Optional[str] = None

        # 流式速度控制的持久化状态
        self._velocity_controller: Optional[object] = None
        self._velocity_ctx: Optional[object] = None
        self._velocity_control_thread: Optional[threading.Thread] = None
        self._velocity_control_stop = threading.Event()
        self._velocity_control_lock = threading.Lock()
        self._velocity_accel_time: float = 0.1  # 加速度时间
        self._velocity_control_freq: float = 200.0  # 控制频率
        self._velocity_time_step: float = 1.0 / 200.0  # 积分时间步
        self._last_velocity_target: Optional[np.ndarray] = None  # 上一次的速度目标
        self._velocity_ramp_start_time: Optional[float] = None  # 梯形斜坡开始时间

    def move_to_start(self) -> None:
        if self.auto_set_default_behavior:
            self.panda.set_default_behavior()
        if self.auto_move_to_start:
            self.panda.move_to_start()
        self.T_0 = SE3(self.panda.get_pose(), check=False)

    def move_to_pose(self, poses: list, cart_speed_factor=None, stiffness=None) -> None:
        """poses传的是世界坐标系的坐标,格式: [x, y, z, roll, pitch, yaw]"""
        initial_pose = SE3(self.panda.get_pose(), check=False)

        se3_poses = []
        for pose in poses:
            trans = SE3.Trans(pose[0], pose[1], pose[2])
            if len(pose) == 6:
                rot = SE3.RPY([pose[3], pose[4], pose[5]], order="xyz")
            else:
                rot = SE3.Rt(initial_pose.R, t=[0, 0, 0], check=False)
            se3_poses.append(trans * rot)

        if cart_speed_factor is None:
            cart_speed_factor = self.cart_speed_factor
        if stiffness is None:
            stiffness = self.stiffness
        self.panda.move_to_pose(se3_poses, speed_factor=cart_speed_factor)
        self.T_0 = SE3(self.panda.get_pose(), check=False)

    def move_to_joint_position(
        self, poses: list, joint_speed_factor=None, stiffness=None
    ) -> None:
        """poses传的是世界坐标系的坐标,格式: [x, y, z, roll, pitch, yaw]"""
        initial_pose = SE3(self.panda.get_pose(), check=False)
        initial_rpy = initial_pose.rpy(order="xyz", unit="rad")

        def _normalize_angle(angle: float) -> float:
            return (angle + np.pi) % (2 * np.pi) - np.pi

        q_current = self.panda.q
        current_yaw = initial_rpy[2]

        qs = []
        for i, pose in enumerate(poses):
            x, y, z = pose[:3]
            trans = SE3.Trans(x, y, z)

            if len(pose) == 6:
                roll, pitch, yaw = pose[3:6]
                yaw_candidates = [yaw]
            else:
                roll, pitch = initial_rpy[0], initial_rpy[1]
                base_candidates = [current_yaw]
                offsets = [np.pi / 6, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
                for offset in offsets:
                    base_candidates.append(_normalize_angle(current_yaw + offset))
                    base_candidates.append(_normalize_angle(current_yaw - offset))
                yaw_candidates = []
                seen = set()
                for candidate in base_candidates:
                    key = round(candidate, 6)
                    if key not in seen:
                        seen.add(key)
                        yaw_candidates.append(candidate)

            best_q = None
            min_distance = float("inf")
            chosen_yaw = current_yaw

            for yaw_candidate in yaw_candidates:
                rot = SE3.RPY([roll, pitch, yaw_candidate], order="xyz")
                T = trans * rot
                all_solutions = panda_py.ik_full(T, q_init=q_current)
                for solution in all_solutions:
                    if np.any(np.isnan(solution)):
                        continue
                    distance = np.linalg.norm(solution - q_current)
                    if distance < min_distance:
                        min_distance = distance
                        best_q = solution
                        chosen_yaw = yaw_candidate

            if best_q is None:
                raise ValueError(
                    f"IK求解失败:目标位姿 {i} 无解或超出工作空间。\n"
                    f"目标位置: x={x:.3f}, y={y:.3f}, z={z:.3f}\n"
                    f"请检查目标位置是否在机械臂可达范围内。"
                )
            qs.append(best_q)
            q_current = best_q
            if len(pose) == 6:
                current_yaw = yaw
            else:
                current_yaw = chosen_yaw
        if joint_speed_factor is None:
            joint_speed_factor = self.joint_speed_factor
        if stiffness is None:
            stiffness = self.stiffness
        self.panda.move_to_joint_position(qs, speed_factor=joint_speed_factor)
        self.T_0 = SE3(self.panda.get_pose(), check=False)

    def gripper_grasp(
        self,
        width: float = 0.0,
        speed: float = 0.2,
        force: float = 10.0,
        epsilon_inner: float = 0.04,
        epsilon_outer: float = 0.04,
    ) -> bool:
        return self.gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer)

    def gripper_move(self, width: float = 0.08, speed: float = 0.2) -> bool:
        return self.gripper.move(width, speed)

    def gripper_open(self, width: float = 0.05, speed: float = 0.2) -> bool:
        return self.gripper_move(width=width, speed=speed)

    def gripper_close(
        self,
        width: float = 0.0,
        speed: float = 0.2,
        force: float = 60.0,
        epsilon_inner: float = 0.04,
        epsilon_outer: float = 0.04,
    ) -> bool:
        return self.gripper_grasp(width, speed, force, epsilon_inner, epsilon_outer)

    def move_delta_pose(
        self,
        action: np.ndarray,
        joint_speed_factor: float = None,
        use_cartesian: bool = True,
        flip_axes: tuple = (False, False, False),  # (flip_x, flip_y, flip_z)
        delta_frame: Literal["ee", "base"] = "ee",
    ) -> None:
        """
        执行增量位姿移动（GA-DDPG 模型输出的 6D delta action）

        Args:
            action: 6D 增量向量 [dx, dy, dz, roll, pitch, yaw]，在末端坐标系下
            joint_speed_factor: 可选的速度因子
            use_cartesian: 是否使用笛卡尔空间移动（推荐True，避免关节大幅转动）
            flip_axes: 翻转哪些轴 (x, y, z)，用于坐标系对齐
            delta_frame: 增量动作所在坐标系；"ee" 表示末端坐标系右乘，"base" 表示基座/世界坐标系左乘
        """
        if any(flip_axes):
            action = action.copy()
            for i, flip in enumerate(flip_axes[:3]):
                if flip:
                    action[i] = -action[i]
            if flip_axes != (False, False, False):
                print(
                    f"[DEBUG] Flipped action: flip_axes={flip_axes}, result=[{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]"
                )

        current_pose = self.ee_pose_matrix
        delta_pose = self._unpack_action(action)
        # 计算目标位姿：
        # - 末端坐标系增量：target = current @ delta (右乘)
        # - 基座/世界坐标系增量：target = delta @ current (左乘)
        if delta_frame == "ee":
            target_pose = current_pose @ delta_pose
        elif delta_frame == "base":
            target_pose = delta_pose @ current_pose
        else:
            raise ValueError(f"delta_frame must be 'ee' or 'base', got {delta_frame!r}")

        if use_cartesian:
            cart_speed = (
                joint_speed_factor if joint_speed_factor else self.cart_speed_factor
            )
            self.panda.move_to_pose(target_pose, speed_factor=cart_speed)
            self.T_0 = SE3(self.panda.get_pose(), check=False)
        else:
            target_se3 = SE3(target_pose, check=False)
            position = target_se3.t
            rpy = target_se3.rpy(order="xyz", unit="rad")
            target_6d = [
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(rpy[0]),
                float(rpy[1]),
                float(rpy[2]),
            ]
            self.move_to_joint_position(
                [target_6d], joint_speed_factor=joint_speed_factor
            )

    def _unpack_action(self, action: np.ndarray) -> np.ndarray:
        from transforms3d.euler import euler2mat

        action = np.asarray(action, dtype=np.float64).flatten()
        if action.shape[0] != 6:
            raise ValueError(f"action must be 6D, got shape {action.shape}")

        pose_delta = np.eye(4, dtype=np.float32)
        pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
        pose_delta[:3, 3] = action[:3]

        return pose_delta

    def apply_joint_velocity(
        self,
        joint_velocity: np.ndarray,
        duration: float = 0.01,
        acceleration_time: float = 0.01,
        *,
        control_frequency: float = 200.0,
        max_abs_velocity: float | None = None,
        streaming: bool = False,
    ) -> None:
        """
        关节速度控制 - 支持单次执行或流式持续控制

        Args:
            joint_velocity: 7维关节速度 (rad/s)
            duration: 执行时长（秒），仅在 streaming=False 时有效
            acceleration_time: 加速时间（秒）
            control_frequency: 控制频率 (Hz)，默认 200Hz
            max_abs_velocity: 最大速度限幅（可选）
            streaming:
                - False (默认): 单次执行，立即返回
                - True: 启动持久化控制器，后续调用只需更新速度命令

        使用方式:
            # 单次执行（原来的用法）
            arm.apply_joint_velocity(velocity, duration=0.066)

            # 流式控制（高频调用）
            arm.start_velocity_streaming(control_frequency=200.0)
            for _ in range(1000):
                arm.apply_joint_velocity(velocity, streaming=True)
            arm.stop_velocity_streaming()
        """
        joint_velocity = np.asarray(joint_velocity, dtype=np.float64).flatten()
        if joint_velocity.shape[0] != 7:
            raise ValueError(
                f"Expected 7-DoF joint velocity, got {joint_velocity.shape}"
            )

        if max_abs_velocity is not None:
            joint_velocity = np.clip(
                joint_velocity, -max_abs_velocity, max_abs_velocity
            )

        if streaming:
            # 流式模式：只更新控制命令
            self._update_velocity_command(joint_velocity)
        else:
            # 单次执行模式
            self._execute_velocity_once(
                joint_velocity, duration, acceleration_time, control_frequency
            )

    def _execute_velocity_once(
        self,
        joint_velocity: np.ndarray,
        duration: float,
        acceleration_time: float,
        control_frequency: float,
    ) -> None:
        """单次执行速度控制（不持久化）"""
        from panda_py import controllers

        # 防止已有控制器冲突（不影响持久化的流式控制器）
        if self._velocity_controller is None:
            self.panda.stop_controller()

        ctrl = controllers.IntegratedVelocity()
        self.panda.start_controller(ctrl)

        duration = max(0.0, float(duration))
        accel = max(0.0, float(acceleration_time))

        try:
            with self.panda.create_context(
                frequency=control_frequency, max_runtime=duration
            ) as ctx:
                while ctx.ok():
                    t = ctx.time
                    if accel > 0.0 and duration > 0.0:
                        # 梯形速度曲线：起始加速、末尾减速
                        ramp_up = min(1.0, t / accel)
                        ramp_down = min(1.0, max(0.0, (duration - t) / accel))
                        scale = min(ramp_up, ramp_down)
                    else:
                        scale = 1.0

                    ctrl.set_control(joint_velocity * scale)
        finally:
            # 结束时清零
            ctrl.set_control(np.zeros(7))
            self.panda.stop_controller()

    def start_velocity_streaming(
        self,
        control_frequency: float = 200.0,
        acceleration_time: float = 0.1,
        time_step: float | None = None,
    ) -> None:
        """
        启动持久化速度控制器（用于高频流式调用）

        Args:
            control_frequency: 控制循环频率 (Hz)
            acceleration_time: 加速/减速时间（秒），用于梯形速度曲线
            time_step: 积分时间步 (秒)。如果提供，每个速度命令会乘以这个值来匹配原来的积分行为
                      例如：原来 duration=0.066s，现在应该传 time_step=0.066

        Example:
            arm.start_velocity_streaming(control_frequency=200.0, time_step=0.066)
            try:
                for _ in range(1000):
                    arm.apply_joint_velocity(vel, streaming=True)
            finally:
                arm.stop_velocity_streaming()
        """
        if self._velocity_controller is not None:
            print("[Warning] Velocity streaming already started, stopping first...")
            self.stop_velocity_streaming()

        from panda_py import controllers

        self.panda.stop_controller()
        self._velocity_controller = controllers.IntegratedVelocity()
        self.panda.start_controller(self._velocity_controller)

        # 初始化为零速度
        self._velocity_controller.set_control(np.zeros(7))

        # 保存加速度参数和时间步
        self._velocity_accel_time = max(0.0, float(acceleration_time))
        self._velocity_control_freq = max(1.0, float(control_frequency))
        self._velocity_time_step = (
            float(time_step)
            if time_step is not None
            else (1.0 / self._velocity_control_freq)
        )
        self._last_velocity_target = np.zeros(7)
        self._velocity_ramp_start_time = None
        self._velocity_control_stop.clear()

        print(
            f"[Velocity] 已启动持续速度控制器 (freq={control_frequency:.0f}Hz, accel_time={acceleration_time:.3f}s, time_step={self._velocity_time_step:.4f}s)"
        )

    def _update_velocity_command(self, joint_velocity: np.ndarray) -> None:
        """在流式模式下更新速度命令"""
        if self._velocity_controller is None:
            raise RuntimeError(
                "Velocity controller not started. Call start_velocity_streaming() first or use streaming=False."
            )

        # 应用时间步缩放，保持原来的积分行为
        # 原来：target_q = current_q + joint_velocity * duration
        # 现在：速度 * time_step 相当于原来的位置增量
        scaled_velocity = joint_velocity * self._velocity_time_step

        with self._velocity_control_lock:
            try:
                self._velocity_controller.set_control(scaled_velocity)
            except Exception as e:
                print(f"[Error] Failed to set velocity command: {e}")
                raise

    def set_velocity_command(self, joint_velocity: np.ndarray) -> None:
        """
        快速设置速度命令（不加锁，性能更高，用于流式控制）

        需要先调用 start_velocity_streaming()
        """
        joint_velocity = np.asarray(joint_velocity, dtype=np.float64).flatten()
        if joint_velocity.shape[0] != 7:
            raise ValueError(
                f"Expected 7-DoF joint velocity, got {joint_velocity.shape}"
            )

        if self._velocity_controller is None:
            raise RuntimeError(
                "Velocity controller not started. Call start_velocity_streaming() first."
            )

        self._velocity_controller.set_control(joint_velocity)

    def stop_velocity_streaming(self) -> None:
        """停止持久化速度控制器"""
        with self._velocity_control_lock:
            if self._velocity_controller is None:
                return

            try:
                # 清零速度
                self._velocity_controller.set_control(np.zeros(7))
                self.panda.stop_controller()
                self._velocity_controller = None
                print("[Velocity] 已停止持续速度控制器")
            except Exception as e:
                print(f"[Error] Error stopping velocity controller: {e}")
                raise

    @property
    def state(self):
        """
        获取机器人状态。

        注意：当流式速度控制器运行时，使用 get_state() 返回缓存快照，
        而不是 read_once() 来避免阻塞冲突。
        """
        if self._velocity_controller is not None:
            # 流式控制运行中，用缓存状态
            robot_state = self.panda.get_state()
        else:
            # 无控制器运行，可以直接读取最新状态
            robot_state = self.panda.get_robot().read_once()

        return {
            "joint_positions": robot_state.q,
            "joint_velocities": robot_state.dq,
            "end_effector_pose": robot_state.O_T_EE,
            "joint_torques": robot_state.tau_J,
            "cartesian_position": (
                robot_state.O_T_EE[:3, 3]
                if hasattr(robot_state.O_T_EE, "shape")
                else np.zeros(3)
            ),
            "gripper_position": (
                self.gripper.read_once().width if hasattr(self, "gripper") else 0.0
            ),
        }

    @property
    def ee_pose_matrix(self):
        robot_state = self.panda.get_robot().read_once()
        pose_matrix = np.array(robot_state.O_T_EE).reshape(4, 4).T
        return pose_matrix.astype(np.float32)

    @property
    def pose(self):
        robot_state = self.panda.get_robot().read_once()
        pose_matrix = np.array(robot_state.O_T_EE).reshape(4, 4).T
        T = SE3(pose_matrix, check=False)
        position = T.t
        rpy = T.rpy(order="xyz")
        return np.concatenate([position, rpy])

    def __del__(self):
        self.cleanup()

    def cleanup(self) -> None:
        with suppress(Exception):
            self.panda.disable_logging()
        with suppress(Exception):
            self.panda.stop_controller()
        with suppress(Exception):
            self.desk.stop_listen()
        with suppress(Exception):
            self.desk.deactivate_fci()
        with suppress(Exception):
            self.desk.release_control()
        with suppress(Exception):
            self.desk.logout()




if __name__ == "__main__":
    robotic_arm_instance = RoboticArmControler()
    try:
        print("")
        robotic_arm_instance.move_to_start()
        robotic_arm_instance.gripper_grasp(width=0.5)
    finally:
        robotic_arm_instance.cleanup()
