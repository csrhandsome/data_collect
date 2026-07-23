# 模块加载更快
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
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


RealtimeControlMode = Literal["joint_position"]


@dataclass
class _GripperCommand:
    kind: Literal["move", "grasp"]
    params: dict[str, float]
    done: threading.Event
    result: Optional[bool] = None
    error: Optional[BaseException] = None


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
        realtime_control: bool = False,
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

        self.realtime_control = bool(realtime_control)

        # 最小实时控制状态：仅在 realtime_control=True 时启用。
        self._realtime_controller: Optional[object] = None
        self._realtime_paused: Optional[threading.Event] = None
        self._realtime_lock: Optional[threading.Lock] = None
        self._joint_position_target: Optional[np.ndarray] = None
        self._joint_velocity_feedforward: Optional[np.ndarray] = None
        self._velocity_time_step: Optional[float] = None
        if self.realtime_control:
            self._realtime_paused = threading.Event()
            self._realtime_lock = threading.Lock()
            self._joint_velocity_feedforward = np.zeros(7, dtype=np.float64)
            self._velocity_time_step = 1.0 / 200.0

        # 夹爪后台线程
        self._gripper_lock = threading.Lock()
        self._gripper_command: Optional[_GripperCommand] = None
        self._gripper_active_command: Optional[_GripperCommand] = None
        self._gripper_request = threading.Event()
        self._gripper_stop = threading.Event()
        self._gripper_thread = threading.Thread(
            target=self._gripper_worker,
            name="franka-gripper-worker",
            daemon=True,
        )
        self._gripper_thread.start()

    def move_to_start(self) -> None:
        if self.auto_set_default_behavior:
            self.panda.set_default_behavior()
        if self.auto_move_to_start:
            self.panda.move_to_start()
        self.T_0 = SE3(self.panda.get_pose(), check=False)

    def wait_until_stopped(
        self,
        *,
        timeout_s: float = 2.0,
        velocity_tol: float = 5e-3,
        poll_s: float = 0.01,
    ) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            qvel = np.asarray(self.panda.get_state().dq, dtype=np.float64)
            if np.max(np.abs(qvel)) <= velocity_tol:
                return True
            time.sleep(poll_s)
        return False

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

    @staticmethod
    def _as_joint_vector(values: Sequence[float], *, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64).flatten()
        if vector.shape != (7,):
            raise ValueError(f"{name} must be 7D, got shape {vector.shape}")
        return vector

    def _get_robot_state_snapshot(self):
        if self._realtime_controller is not None:
            return self.panda.get_state()
        return self.panda.get_robot().read_once()

    def _require_realtime_control(self) -> None:
        if (
            not self.realtime_control
            or self._realtime_lock is None
            or self._realtime_paused is None
            or self._joint_velocity_feedforward is None
            or self._velocity_time_step is None
        ):
            raise RuntimeError(
                "Realtime control is disabled for this arm. Create `RoboticArmControler(..., realtime_control=True)` to use realtime APIs."
            )

    def _solve_joint_targets_from_poses(
        self, poses: Sequence[Sequence[float]]
    ) -> list[np.ndarray]:
        initial_pose = SE3(self.panda.get_pose(), check=False)
        initial_rpy = initial_pose.rpy(order="xyz", unit="rad")

        def _normalize_angle(angle: float) -> float:
            return (angle + np.pi) % (2 * np.pi) - np.pi

        robot_state = self._get_robot_state_snapshot()
        q_current = np.asarray(robot_state.q, dtype=np.float64)
        current_yaw = float(initial_rpy[2])

        qs = []
        for i, pose in enumerate(poses):
            pose = np.asarray(pose, dtype=np.float64).flatten()
            if pose.shape[0] not in (3, 6):
                raise ValueError(f"pose[{i}] must be 3D or 6D, got shape {pose.shape}")

            x, y, z = pose[:3]
            trans = SE3.Trans(float(x), float(y), float(z))

            if pose.shape[0] == 6:
                roll, pitch, yaw = pose[3:6]
                yaw_candidates = [float(yaw)]
            else:
                roll, pitch = float(initial_rpy[0]), float(initial_rpy[1])
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
                target_pose = trans * rot
                all_solutions = panda_py.ik_full(target_pose, q_init=q_current)
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
                    "请检查目标位置是否在机械臂可达范围内。"
                )

            best_q = np.asarray(best_q, dtype=np.float64)
            qs.append(best_q)
            q_current = best_q
            current_yaw = float(yaw if pose.shape[0] == 6 else chosen_yaw)

        return qs

    def move_to_joint_position(
        self, poses: list, joint_speed_factor=None, stiffness=None
    ) -> None:
        """poses传的是世界坐标系的坐标,格式: [x, y, z, roll, pitch, yaw]"""
        qs = self._solve_joint_targets_from_poses(poses)
        if joint_speed_factor is None:
            joint_speed_factor = self.joint_speed_factor
        if stiffness is None:
            stiffness = self.stiffness
        self.panda.move_to_joint_position(qs, speed_factor=joint_speed_factor)
        self.T_0 = SE3(self.panda.get_pose(), check=False)

    def _wait_for_gripper_command(
        self, command: _GripperCommand, timeout: Optional[float] = None
    ) -> bool:
        if not command.done.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for gripper command to finish")
        if command.error is not None:
            raise RuntimeError("Gripper command failed") from command.error
        return bool(command.result)

    def _submit_gripper_command(
        self,
        kind: Literal["move", "grasp"],
        params: dict[str, float],
        *,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        command = _GripperCommand(
            kind=kind,
            params={key: float(value) for key, value in params.items()},
            done=threading.Event(),
        )
        with self._gripper_lock:
            if (
                self._gripper_command is not None
                or self._gripper_active_command is not None
            ):
                raise RuntimeError(
                    "Gripper is busy. Wait for the previous gripper command to finish first."
                )
            self._gripper_command = command
            self._gripper_request.set()

        if not wait:
            return True
        return self._wait_for_gripper_command(command, timeout=timeout)

    def _pause_realtime_controller(self, timeout: float = 2.0) -> bool:
        del timeout
        if (
            not self.realtime_control
            or self._realtime_lock is None
            or self._realtime_paused is None
        ):
            return False
        with self._realtime_lock:
            controller = self._realtime_controller
            if controller is None:
                return False

            self._hold_current_joint_position_with_controller(controller)
            self._realtime_paused.set()
            self.panda.stop_controller()
            return True

    def _resume_realtime_controller(self, timeout: float = 2.0) -> None:
        del timeout
        if (
            not self.realtime_control
            or self._realtime_lock is None
            or self._realtime_paused is None
        ):
            return
        with self._realtime_lock:
            controller = self._realtime_controller
            if controller is None:
                return

            self.panda.start_controller(controller)
            if self._joint_position_target is None:
                qpos = self._hold_current_joint_position_with_controller(controller)
                self._joint_position_target = qpos.copy()
                self._joint_velocity_feedforward = np.zeros(7, dtype=np.float64)
            else:
                controller.set_control(
                    self._joint_position_target,
                    self._joint_velocity_feedforward,
                )
            self._realtime_paused.clear()

    def _execute_gripper_command(self, command: _GripperCommand) -> bool:
        if command.kind == "move":
            return self.gripper.move(command.params["width"], command.params["speed"])
        return self.gripper.grasp(
            command.params["width"],
            command.params["speed"],
            command.params["force"],
            command.params["epsilon_inner"],
            command.params["epsilon_outer"],
        )

    def _gripper_worker(self) -> None:
        while not self._gripper_stop.is_set():
            self._gripper_request.wait(0.05)
            if self._gripper_stop.is_set():
                break
            if not self._gripper_request.is_set():
                continue

            with self._gripper_lock:
                command = self._gripper_command
                self._gripper_command = None
                self._gripper_active_command = command
                self._gripper_request.clear()

            if command is None:
                continue

            paused = False
            try:
                paused = self._pause_realtime_controller()
                command.result = self._execute_gripper_command(command)
            except BaseException as exc:
                command.error = exc
            finally:
                if paused:
                    try:
                        self._resume_realtime_controller()
                    except BaseException as exc:
                        if command.error is None:
                            command.error = exc
                with self._gripper_lock:
                    self._gripper_active_command = None
                command.done.set()

    @property
    def gripper_busy(self) -> bool:
        with self._gripper_lock:
            command = self._gripper_command
            active = self._gripper_active_command
        return (
            (command is not None and not command.done.is_set())
            or (active is not None and not active.done.is_set())
            or self._gripper_request.is_set()
        )

    def wait_gripper(self, timeout: Optional[float] = None) -> None:
        with self._gripper_lock:
            command = self._gripper_command
            active = self._gripper_active_command
        if command is None:
            command = active
        if command is None:
            return
        self._wait_for_gripper_command(command, timeout=timeout)

    def gripper_grasp(
        self,
        width: float = 0.0,
        speed: float = 0.2,
        force: float = 10.0,
        epsilon_inner: float = 0.04,
        epsilon_outer: float = 0.04,
        *,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        return self._submit_gripper_command(
            "grasp",
            {
                "width": width,
                "speed": speed,
                "force": force,
                "epsilon_inner": epsilon_inner,
                "epsilon_outer": epsilon_outer,
            },
            wait=wait,
            timeout=timeout,
        )

    def gripper_move(
        self,
        width: float = 0.08,
        speed: float = 0.2,
        *,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        return self._submit_gripper_command(
            "move",
            {"width": width, "speed": speed},
            wait=wait,
            timeout=timeout,
        )

    def gripper_open(
        self,
        width: float = 0.05,
        speed: float = 0.2,
        *,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        return self.gripper_move(width=width, speed=speed, wait=wait, timeout=timeout)

    def safe_open(
        self,
        width: float = 0.05,
        speed: float = 0.2,
        *,
        release_height: Optional[float] = None,
        joint_speed_factor: Optional[float] = None,
        descent_deadband: float = 5e-3,
        settle_timeout_s: float = 2.0,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        """Lower the tool to a safer release height before opening the gripper.

        The descent prefers the controller's existing pose IK path:
        1. Keep the current full 6D tool pose and only lower z.
        2. If that IK fails, fall back to position-only IK so yaw can adjust.
        """
        target_z = self.default_height if release_height is None else release_height
        if target_z is not None:
            target_z = float(target_z)
            if target_z < 0.0:
                raise ValueError(f"release_height must be >= 0, got {target_z:.6f}")

            current_pose = np.asarray(self.pose, dtype=np.float64)
            current_z = float(current_pose[2])
            if current_z - target_z > float(descent_deadband):
                target_pose = current_pose.copy()
                target_pose[2] = target_z
                move_error: Optional[Exception] = None

                try:
                    self.move_to_joint_position(
                        [target_pose.tolist()],
                        joint_speed_factor=joint_speed_factor,
                    )
                    self.wait_until_stopped(timeout_s=settle_timeout_s)
                except Exception as exc:
                    move_error = exc

                if move_error is not None:
                    try:
                        self.move_to_joint_position(
                            [target_pose[:3].tolist()],
                            joint_speed_factor=joint_speed_factor,
                        )
                        self.wait_until_stopped(timeout_s=settle_timeout_s)
                        move_error = None
                    except Exception as fallback_exc:
                        move_error = fallback_exc

                if move_error is not None:
                    print(
                        "[Warning] safe_open failed to lower the end effector "
                        f"from z={current_z:.3f} to z={target_z:.3f}; "
                        f"opening in place instead. error={move_error}"
                    )

        return self.gripper_open(width=width, speed=speed, wait=wait, timeout=timeout)

    def gripper_close(
        self,
        width: float = 0.0,
        speed: float = 0.2,
        force: float = 60.0,
        epsilon_inner: float = 0.04,
        epsilon_outer: float = 0.04,
        *,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        return self.gripper_grasp(
            width,
            speed,
            force,
            epsilon_inner,
            epsilon_outer,
            wait=wait,
            timeout=timeout,
        )

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

    def _hold_current_joint_position_with_controller(self, controller) -> np.ndarray:
        robot_state = self.panda.get_state()
        qpos = np.asarray(robot_state.q, dtype=np.float64)
        controller.set_control(qpos, np.zeros(7, dtype=np.float64))
        return qpos

    @property
    def realtime_mode(self) -> Optional[RealtimeControlMode]:
        if self._realtime_controller is None:
            return None
        return "joint_position"

    def start_joint_position_streaming(
        self,
        control_frequency: float = 200.0,
        settle_s: float = 0.5,
    ) -> None:
        self._require_realtime_control()
        from panda_py import controllers

        controller = controllers.JointPosition()
        self.stop_realtime_control(raise_on_timeout=False)
        self.panda.stop_controller()
        self.panda.start_controller(controller)

        if settle_s > 0.0:
            time.sleep(max(0.0, float(settle_s)))
        qpos = self._hold_current_joint_position_with_controller(controller)
        with self._realtime_lock:
            self._realtime_controller = controller
            self._joint_position_target = qpos.copy()
            self._joint_velocity_feedforward = np.zeros(7, dtype=np.float64)
            self._velocity_time_step = 1.0 / max(1.0, float(control_frequency))
            self._realtime_paused.clear()

    def hold_current_joint_position(self) -> np.ndarray:
        self._require_realtime_control()
        robot_state = self._get_robot_state_snapshot()
        qpos = np.asarray(robot_state.q, dtype=np.float64)
        with self._realtime_lock:
            self._joint_position_target = qpos.copy()
            self._joint_velocity_feedforward = np.zeros(7, dtype=np.float64)
            controller = self._realtime_controller
            if controller is not None and not self._realtime_paused.is_set():
                controller.set_control(
                    self._joint_position_target,
                    self._joint_velocity_feedforward,
                )
        return qpos

    def set_joint_position_command(
        self,
        joint_position: Sequence[float],
        joint_velocity: Optional[Sequence[float]] = None,
    ) -> None:
        self._require_realtime_control()
        controller = self._realtime_controller
        if controller is None:
            raise RuntimeError(
                "Realtime joint controller not started. Call start_joint_position_streaming() first, then manage the loop with `with arm.panda.create_context(...):`."
            )

        qpos = self._as_joint_vector(joint_position, name="joint_position")
        qvel = (
            np.zeros(7, dtype=np.float64)
            if joint_velocity is None
            else self._as_joint_vector(joint_velocity, name="joint_velocity")
        )

        with self._realtime_lock:
            self._joint_position_target = qpos
            self._joint_velocity_feedforward = qvel
            if (
                not self._realtime_paused.is_set()
                and self._realtime_controller is not None
            ):
                self._realtime_controller.set_control(qpos, qvel)

    def set_pose_command(
        self,
        pose: Sequence[float],
        joint_velocity: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        target_q = self._solve_joint_targets_from_poses([pose])[-1]
        self.set_joint_position_command(target_q, joint_velocity=joint_velocity)
        return target_q

    def move_delta_pose_realtime(
        self,
        action: np.ndarray,
        joint_velocity: Optional[Sequence[float]] = None,
        flip_axes: tuple = (False, False, False),
        delta_frame: Literal["ee", "base"] = "ee",
    ) -> np.ndarray:
        if any(flip_axes):
            action = np.asarray(action, dtype=np.float64).copy()
            for i, flip in enumerate(flip_axes[:3]):
                if flip:
                    action[i] = -action[i]

        current_pose = self.ee_pose_matrix
        delta_pose = self._unpack_action(np.asarray(action, dtype=np.float64))
        if delta_frame == "ee":
            target_pose = current_pose @ delta_pose
        elif delta_frame == "base":
            target_pose = delta_pose @ current_pose
        else:
            raise ValueError(f"delta_frame must be 'ee' or 'base', got {delta_frame!r}")

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
        return self.set_pose_command(target_6d, joint_velocity=joint_velocity)

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
        joint_velocity = self._as_joint_vector(joint_velocity, name="joint_velocity")

        if max_abs_velocity is not None:
            joint_velocity = np.clip(
                joint_velocity, -float(max_abs_velocity), float(max_abs_velocity)
            )

        if streaming:
            self._update_velocity_command(joint_velocity)
        else:
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
        from panda_py import controllers

        ctrl = controllers.IntegratedVelocity()
        duration = max(0.0, float(duration))
        accel = max(0.0, float(acceleration_time))

        try:
            self.panda.stop_controller()
            self.panda.start_controller(ctrl)
            with self.panda.create_context(
                frequency=control_frequency, max_runtime=duration
            ) as ctx:
                while ctx.ok():
                    t = ctx.time
                    if accel > 0.0 and duration > 0.0:
                        ramp_up = min(1.0, t / accel)
                        ramp_down = min(1.0, max(0.0, (duration - t) / accel))
                        scale = min(ramp_up, ramp_down)
                    else:
                        scale = 1.0
                    ctrl.set_control(joint_velocity * scale)
        finally:
            with suppress(Exception):
                ctrl.set_control(np.zeros(7, dtype=np.float64))
            with suppress(Exception):
                self.panda.stop_controller()

    def start_velocity_streaming(
        self,
        control_frequency: float = 200.0,
        acceleration_time: float = 0.1,
        time_step: float | None = None,
    ) -> None:
        self._require_realtime_control()
        del acceleration_time
        self._velocity_time_step = (
            float(time_step)
            if time_step is not None
            else 1.0 / max(1.0, float(control_frequency))
        )
        self.start_joint_position_streaming(
            control_frequency=control_frequency,
            settle_s=0.0,
        )

    def _update_velocity_command(self, joint_velocity: np.ndarray) -> None:
        self._require_realtime_control()
        if self._realtime_controller is None:
            raise RuntimeError(
                "Realtime joint controller not started. Call start_joint_position_streaming() first, then manage the loop with `with arm.panda.create_context(...):`."
            )

        current_q = np.asarray(self.panda.get_state().q, dtype=np.float64)
        target_q = current_q + self._as_joint_vector(
            joint_velocity * float(self._velocity_time_step),
            name="joint_velocity",
        )
        self.set_joint_position_command(target_q, joint_velocity=joint_velocity)

    def set_velocity_command(self, joint_velocity: np.ndarray) -> None:
        self._update_velocity_command(joint_velocity)

    def stop_realtime_control(
        self,
        *,
        timeout: float = 2.0,
        raise_on_timeout: bool = True,
    ) -> None:
        del timeout, raise_on_timeout
        if (
            not self.realtime_control
            or self._realtime_lock is None
            or self._realtime_paused is None
        ):
            return
        with self._realtime_lock:
            controller = self._realtime_controller
            if controller is None:
                return

            with suppress(Exception):
                self._hold_current_joint_position_with_controller(controller)
            with suppress(Exception):
                self.panda.stop_controller()

            self._realtime_controller = None
            self._joint_position_target = None
            self._joint_velocity_feedforward = np.zeros(7, dtype=np.float64)
            self._velocity_time_step = 1.0 / 200.0
            self._realtime_paused.clear()

    def stop_velocity_streaming(self) -> None:
        self.stop_realtime_control()

    def stop_joint_position_streaming(self) -> None:
        self.stop_realtime_control()

    @property
    def state(self):
        """
        获取机器人状态。

        注意：当实时关节控制运行时，使用 get_state() 返回缓存快照，
        而不是 read_once() 来避免阻塞冲突。
        """
        robot_state = self._get_robot_state_snapshot()

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
        robot_state = self._get_robot_state_snapshot()
        pose_matrix = np.array(robot_state.O_T_EE).reshape(4, 4).T
        return pose_matrix.astype(np.float32)

    @property
    def pose(self):
        robot_state = self._get_robot_state_snapshot()
        pose_matrix = np.array(robot_state.O_T_EE).reshape(4, 4).T
        T = SE3(pose_matrix, check=False)
        position = T.t
        rpy = T.rpy(order="xyz")
        return np.concatenate([position, rpy])

    def __del__(self):
        self.cleanup()

    def cleanup(self) -> None:
        with suppress(Exception):
            self.wait_gripper(timeout=2.0)
        with suppress(Exception):
            self.stop_realtime_control(raise_on_timeout=False)
        with suppress(Exception):
            self._gripper_stop.set()
            self._gripper_request.set()
        with suppress(Exception):
            if self._gripper_thread.is_alive():
                self._gripper_thread.join(timeout=1.0)
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
