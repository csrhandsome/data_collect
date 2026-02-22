#!/usr/bin/env python3
"""
Collect Franka teleop data into a LeRobot dataset using DROID-style keys.

- Cameras: external + wrist (2x RealSense)
- Actions: measured joint velocity (rad/s) + gripper state
- Output: LeRobot dataset (no intermediate HDF5)

uv run xbox_openpi_lerobot_franka_setcontrol.py \
  --instruction "Pick up the brown bottle" \
  --external-camera-serial 825412070292 \
  --wrist-camera-serial 825412070487 \
  --color-only 
"""

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from pygame_gamepad import PygameGamepadTeleop
from realsense_connector import RealSenseConnector
from robotic_arm_controller import RoboticArmControler
from robotic_arm_controller import _camera_capture_worker
from robotic_arm_controller import _LatestFrameBuffer


def _get_gamepad_inputs(teleop: PygameGamepadTeleop) -> tuple[dict, dict]:
    joystick = getattr(teleop, "joystick", None)
    deadzone = float(getattr(teleop, "deadzone", 0.1))

    axes: dict[str, float] = {
        "right_x": 0.0,
        "right_y": 0.0,
        "lt": 0.0,
        "rt": 0.0,
    }
    buttons: dict[str, bool] = {
        "l1": False,
        "r1": False,
        "a": False,
        "b": False,
        "x": False,
        "o": False,
    }

    if joystick is None:
        return axes, buttons

    teleop.update()
    num_axes = joystick.get_numaxes()
    num_buttons = joystick.get_numbuttons()

    def read_axis(index: int, *, apply_deadzone: bool = True) -> float:
        if index < 0 or index >= num_axes:
            return 0.0
        value = float(joystick.get_axis(index))
        if apply_deadzone and abs(value) < deadzone:
            return 0.0
        return value

    def read_button(index: int) -> bool:
        if index < 0 or index >= num_buttons:
            return False
        return bool(joystick.get_button(index))

    if num_axes >= 5:
        axes["right_y"] = read_axis(3)
        axes["right_x"] = read_axis(4)
    elif num_axes >= 4:
        axes["right_x"] = read_axis(2)
        axes["right_y"] = read_axis(3)

    def normalize_trigger(value: float) -> float:
        if value < 0.0:
            return (value + 1.0) / 2.0
        return value

    if num_axes >= 6:
        axes["lt"] = normalize_trigger(read_axis(2, apply_deadzone=False))
        axes["rt"] = normalize_trigger(read_axis(5, apply_deadzone=False))
    elif num_axes >= 5:
        combined = read_axis(2, apply_deadzone=False)
        axes["lt"] = max(0.0, -combined)
        axes["rt"] = max(0.0, combined)

    buttons["l1"] = read_button(4)
    buttons["r1"] = read_button(5)
    buttons["a"] = read_button(0)
    buttons["b"] = read_button(1)
    buttons["x"] = read_button(2)
    buttons["o"] = read_button(3)

    return axes, buttons


def _create_dataset(
    repo_id: str, *, fps: float, image_hw: int, root: Path
) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=float(fps),
        root=root,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (image_hw, image_hw, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (image_hw, image_hw, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (image_hw, image_hw, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=6,
        image_writer_processes=3,
    )


def _load_or_create_dataset(
    repo_id: str, *, fps: float, image_hw: int, root: Path
) -> LeRobotDataset:
    def looks_like_lerobot_dataset(path: Path) -> bool:
        return (path / "meta" / "info.json").is_file() and (
            path / "meta" / "episodes.jsonl"
        ).is_file()

    def resume_existing_dataset_for_recording(path: Path) -> LeRobotDataset:
        """严格续写模式：只从本地加载 metadata，且不做下载/修复。

        目的：避免 LeRobotDataset.__init__ 在发现缺文件时尝试从 Hub 下载。
        """

        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.common.datasets.video_utils import get_safe_default_codec

        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=path)

        # 1) 必须所有已记录 episode 的 parquet 都存在，否则不允许续写。
        missing: list[Path] = []
        for ep_idx in range(meta.total_episodes):
            fpath = meta.root / meta.get_data_file_path(ep_idx)
            if not fpath.is_file():
                missing.append(fpath)
        if missing:
            preview = "\n".join(f"  - {p}" for p in missing[:10])
            more = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
            raise RuntimeError(
                "Cannot resume: dataset is missing episode parquet files:\n"
                f"{preview}{more}\n"
                "Fix the dataset first (e.g. delete/prune incomplete data), then retry."
            )

        # 2) 如果上次录制中断，可能残留 images/episode_{next}. 这种情况会污染续写。
        next_ep = meta.total_episodes
        images_dir = meta.root / "images"
        if images_dir.is_dir():
            leftover = list(images_dir.rglob(f"episode_{next_ep:06d}"))
            if leftover:
                raise RuntimeError(
                    "Cannot resume: found leftover temporary images for the next episode. "
                    f"Please remove '{images_dir}' (or the episode_{next_ep:06d} folder) and retry."
                )

        # 构造一个“录制用”的 LeRobotDataset：hf_dataset 从空开始即可，meta.total_frames 仍会保证 index 连续。
        dataset = LeRobotDataset.__new__(LeRobotDataset)
        dataset.meta = meta
        dataset.repo_id = meta.repo_id
        dataset.root = meta.root
        dataset.revision = None
        dataset.tolerance_s = 1e-4
        dataset.image_writer = None
        dataset.episode_buffer = dataset.create_episode_buffer()
        dataset.episodes = None
        dataset.hf_dataset = dataset.create_hf_dataset()
        dataset.image_transforms = None
        dataset.delta_timestamps = None
        dataset.delta_indices = None
        dataset.episode_data_index = None
        dataset.video_backend = get_safe_default_codec()

        # Async image writer:
        # - Use threads by default to avoid multiprocessing semaphore issues on some systems.
        # - Adjust to processes>0 if your environment supports it and you need higher throughput.
        dataset.start_image_writer(num_processes=0, num_threads=6)
        return dataset

    if root.exists():
        if not root.is_dir():
            raise RuntimeError(f"Dataset path exists and is not a directory: {root}")
        if not looks_like_lerobot_dataset(root):
            raise RuntimeError(
                "Dataset directory exists but doesn't look like a LeRobot dataset "
                f"(missing meta/info.json): {root}"
            )

        try:
            return resume_existing_dataset_for_recording(root)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load existing LeRobot dataset. "
                "Refusing to modify/recreate automatically. "
                "If the last episode is incomplete, prune it manually with: "
                f".venv/bin/python data_analysis/delete_latest_episode.py --dataset {root} --apply"
            ) from exc

    return _create_dataset(repo_id, fps=fps, image_hw=image_hw, root=root)


def _prepare_episode_for_save(dataset: LeRobotDataset) -> None:
    """Align scalar-like features with HF encoding (shape (1,) -> scalar list)."""
    if dataset.episode_buffer is None:
        return
    gripper_values = dataset.episode_buffer.get("gripper_position")
    if not isinstance(gripper_values, list):
        return
    if not gripper_values:
        return
    dataset.episode_buffer["gripper_position"] = [
        float(v.reshape(-1)[0]) if isinstance(v, np.ndarray) else float(v)
        for v in gripper_values
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Franka data into LeRobot (DROID-style keys)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="openpi/franka_droid_lerobot",
        help="Repo id for storing all episodes (all prompts share the same repo).",
    )
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--control-frequency", type=float, default=15.0)
    parser.add_argument(
        "--translation-speed",
        type=float,
        default=0.05,
        help=(
            "Max translation speed. Interpreted as m/s when --speed-units=per_second, "
            "or as m/step when --speed-units=per_step."
        ),
    )
    parser.add_argument(
        "--rotation-speed",
        type=float,
        default=0.6,
        help=(
            "Max rotation speed. Interpreted as rad/s when --speed-units=per_second, "
            "or as rad/step when --speed-units=per_step."
        ),
    )
    parser.add_argument(
        "--speed-units",
        choices=("per_second", "per_step"),
        default="per_second",
        help=(
            "How to interpret --translation-speed/--rotation-speed. "
            "per_second is safer (scaled by dt=1/control_frequency)."
        ),
    )
    parser.add_argument(
        "--max-ee-step-m",
        type=float,
        default=0.004,
        help="Safety clamp for per-loop end-effector translation step (meters).",
    )
    parser.add_argument(
        "--max-position-error-m",
        type=float,
        default=0.03,
        help=(
            "Safety clamp for the distance between commanded target position and current position (meters). "
            "Helps avoid wind-up near singularities."
        ),
    )
    parser.add_argument("--action-epsilon", type=float, default=1e-6)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--color-only",
        action="store_true",
        help="Disable depth stream to reduce USB bandwidth.",
    )
    parser.add_argument(
        "--camera-startup-timeout-s",
        type=float,
        default=10.0,
        help="Seconds to wait for first frames (0 to wait indefinitely).",
    )
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument("--max-duration", type=float, default=3600.0)
    parser.add_argument("--external-camera-serial", type=str, default=None)
    parser.add_argument("--wrist-camera-serial", type=str, default=None)
    parser.add_argument("--image-hw", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, default=0.9)
    parser.add_argument("--no-logging", action="store_true")

    args = parser.parse_args()

    if args.control_frequency <= 0:
        raise ValueError("--control-frequency must be > 0")

    enable_logging = not args.no_logging
    if enable_logging and not args.instruction.strip():
        raise ValueError("--instruction is required when logging is enabled")

    print("=" * 70)
    print("Franka LeRobot data collection (joint velocity actions)")
    print("=" * 70)
    print(f"Control frequency: {args.control_frequency} Hz")
    if args.speed_units == "per_second":
        print(f"Translation speed: {args.translation_speed} m/s")
        print(f"Rotation speed: {args.rotation_speed} rad/s")
    else:
        print(f"Translation speed: {args.translation_speed} m/step")
        print(f"Rotation speed: {args.rotation_speed} rad/step")
    print(f"Action epsilon: {args.action_epsilon}")
    print(
        "Camera stream: "
        f"{args.camera_width}x{args.camera_height}@{args.camera_fps} "
        f"(depth={'off' if args.color_only else 'on'})"
    )
    print("Input device: gamepad")
    print(f"External camera serial: {args.external_camera_serial}")
    print(f"Wrist camera serial: {args.wrist_camera_serial}")
    if enable_logging:
        date = "2_20_1"
        args.repo_id = f"{args.repo_id}_{date}"
        print(f"Logging: enabled (max {args.max_duration} s)")
        print(f"Instruction: {args.instruction}")
        print(f"LeRobot repo_id: {args.repo_id}")
    else:
        print("Logging: disabled")
    print("=" * 70)

    print("Connecting gamepad (pygame) backend...")
    teleop = PygameGamepadTeleop()
    teleop.connect()

    print("Initializing Franka arm...")
    arm = RoboticArmControler()

    print("Initializing RealSense cameras...")
    external_cam = RealSenseConnector(
        serial=args.external_camera_serial,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        enable_depth=not args.color_only,
    )
    wrist_cam = RealSenseConnector(
        serial=args.wrist_camera_serial,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        enable_depth=not args.color_only,
    )
    external_cam.connect()
    wrist_cam.connect()

    dataset = None
    if enable_logging:
        dataset_root = Path(__file__).resolve().parent / "data" / args.repo_id
        resume_existing = dataset_root.exists()
        dataset = _load_or_create_dataset(
            args.repo_id,
            fps=args.control_frequency,
            image_hw=args.image_hw,
            root=dataset_root,
        )
        if resume_existing:
            print(f"LeRobot dataset exists, resuming: {dataset_root}")
        else:
            print(f"LeRobot dataset path: {dataset_root}")

    external_buf = _LatestFrameBuffer()
    wrist_buf = _LatestFrameBuffer()
    external_stop = threading.Event()
    wrist_stop = threading.Event()

    external_thread = threading.Thread(
        target=_camera_capture_worker,
        kwargs={
            "camera": external_cam,
            "buf": external_buf,
            "stop_event": external_stop,
            "timeout_ms": int(args.camera_timeout_ms),
            "crop_scale": float(args.crop_scale),
            "out_hw": int(args.image_hw),
            "label": "external",
        },
        daemon=True,
    )
    wrist_thread = threading.Thread(
        target=_camera_capture_worker,
        kwargs={
            "camera": wrist_cam,
            "buf": wrist_buf,
            "stop_event": wrist_stop,
            "timeout_ms": int(args.camera_timeout_ms),
            "crop_scale": float(args.crop_scale),
            "out_hw": int(args.image_hw),
            "label": "wrist",
        },
        daemon=True,
    )

    external_thread.start()
    wrist_thread.start()

    print("[Camera] Waiting for first frames...")
    wait_start = time.time()
    while external_buf.get_latest() is None or wrist_buf.get_latest() is None:
        elapsed = time.time() - wait_start
        if (
            args.camera_startup_timeout_s > 0
            and elapsed > args.camera_startup_timeout_s
        ):
            if external_buf.get_latest() is None:
                print("  Waiting for external camera frame...")
            if wrist_buf.get_latest() is None:
                print("  Waiting for wrist camera frame...")
            raise RuntimeError(
                "Camera timeout: failed to get first frames within "
                f"{args.camera_startup_timeout_s:.1f}s."
            )
        time.sleep(0.01)
    print("[Camera] First frames acquired, ready to record!")

    print("Opening gripper...")
    arm.gripper_open()
    print("Moving to start position...")
    arm.move_to_start()

    from panda_py import controllers
    from scipy.spatial.transform import Rotation as R

    # 四元数辅助函数 - 统一使用 [x,y,z,w] 格式（panda_py 约定）
    def quat_multiply(q1, q2):
        """四元数乘法，输入输出都是 [x,y,z,w] 格式"""
        r1 = R.from_quat(q1)  # scipy 使用 [x,y,z,w]
        r2 = R.from_quat(q2)
        result = (r1 * r2).as_quat()  # 返回 [x,y,z,w]
        return result

    def euler_to_quat(roll, pitch, yaw):
        """欧拉角转四元数，返回 [x,y,z,w] 格式"""
        r = R.from_euler("xyz", [roll, pitch, yaw])
        return r.as_quat()  # 返回 [x,y,z,w]

    def quat_inverse(q):
        """四元数求逆，输入输出都是 [x,y,z,w] 格式"""
        r = R.from_quat(q)
        return r.inv().as_quat()

    def quat_to_euler(q):
        """四元数转欧拉角，输入 [x,y,z,w] 格式"""
        r = R.from_quat(q)
        return r.as_euler("xyz")

    ctrl = controllers.CartesianImpedance(filter_coeff=1.0)
    arm.panda.start_controller(ctrl)
    time.sleep(1.0)

    target_position = arm.panda.get_position().astype(np.float64)
    target_orientation = arm.panda.get_orientation().astype(np.float64)  # [x,y,z,w]
    gripper_state = 1.0
    last_gripper_cmd = 1.0

    motion_start_threshold = max(float(args.action_epsilon), 1e-3)
    recording_started = False
    recording_started_at: Optional[float] = None
    frame_count = 0
    last_gripper_switch_time = 0.0
    gripper_switch_cooldown_s = 0.12
    reflex_error_occurred = False

    print("\nControl mapping:")
    print("  Left stick: XY translation")
    print("  L2/R2: Z down/up")
    print("  L1/R1: roll rotation")
    print("  Right stick: yaw/pitch rotation")
    print("  A: gripper close | B: gripper open")
    print("  X/O: save episode + reset (press again with no frames to quit)")

    try:
        with arm.panda.create_context(frequency=args.control_frequency) as ctx:
            while ctx.ok():
                if (
                    enable_logging
                    and recording_started
                    and recording_started_at is not None
                    and (time.time() - recording_started_at) > args.max_duration
                ):
                    print("\n[Recording] Max duration reached, stopping...")
                    break

                action = teleop.get_action()
                if not action:
                    time.sleep(0.001)
                    continue

                axes, buttons = _get_gamepad_inputs(teleop)
                if buttons["x"] or buttons["o"]:
                    if enable_logging and frame_count > 0:
                        print(
                            "\n[Control] X/O pressed, saving episode and resetting..."
                        )
                        try:
                            _prepare_episode_for_save(dataset)
                            dataset.save_episode()
                            print(
                                f"[Recording] Saved episode with {frame_count} frames"
                            )
                        except Exception as exc:
                            # NOTE: LeRobotDataset.save_episode mutates episode_buffer (pop size/task).
                            # If it fails halfway, we must clear the buffer, otherwise subsequent saves will
                            # crash with "size key not found".
                            print(f"[Error] Failed to save episode: {exc}")
                            try:
                                dataset.clear_episode_buffer()
                            except Exception:
                                pass
                        finally:
                            frame_count = 0
                            recording_started = False
                            recording_started_at = None

                        arm.panda.stop_controller()
                        arm.gripper_open()
                        gripper_state = 1.0
                        last_gripper_cmd = 1.0
                        arm.move_to_start()

                        target_position = arm.panda.get_position().astype(np.float64)
                        target_orientation = arm.panda.get_orientation().astype(
                            np.float64
                        )  # [x,y,z,w]
                        arm.panda.start_controller(ctrl)
                        continue

                    print("\n[Control] X/O pressed, stopping...")
                    break

                delta = np.zeros(6, dtype=np.float64)

                dt = 1.0 / float(args.control_frequency)
                speed_scale = dt if args.speed_units == "per_second" else 1.0

                delta[0] = (
                    float(action.get("delta_x", 0.0))
                    * args.translation_speed
                    * speed_scale
                )
                delta[1] = (
                    float(action.get("delta_y", 0.0))
                    * args.translation_speed
                    * speed_scale
                )
                delta[2] = (
                    (axes["rt"] - axes["lt"]) * args.translation_speed * speed_scale
                )
                if getattr(teleop, "joystick", None) is None:
                    delta[2] = (
                        float(action.get("delta_z", 0.0))
                        * args.translation_speed
                        * speed_scale
                    )

                if buttons["l1"]:
                    delta[3] -= args.rotation_speed * speed_scale
                if buttons["r1"]:
                    delta[3] += args.rotation_speed * speed_scale
                delta[4] = (-axes["right_y"]) * args.rotation_speed * speed_scale
                delta[5] = (axes["right_x"]) * args.rotation_speed * speed_scale

                gripper_changed = False
                gripper_cmd = last_gripper_cmd
                if buttons["a"]:
                    gripper_cmd = 0.0
                elif buttons["b"]:
                    gripper_cmd = 1.0

                now = time.time()
                if now - last_gripper_switch_time < gripper_switch_cooldown_s:
                    gripper_cmd = last_gripper_cmd

                if gripper_cmd != last_gripper_cmd:
                    gripper_changed = True
                    last_gripper_switch_time = now
                    pos_before = arm.panda.get_position().astype(np.float64)
                    ori_before = arm.panda.get_orientation().astype(
                        np.float64
                    )  # [x,y,z,w]

                    arm.panda.stop_controller()
                    if gripper_cmd > 0.5:
                        arm.gripper_open()
                        gripper_state = 1.0
                    else:
                        arm.gripper_close()
                        gripper_state = 0.0

                    target_position = arm.panda.get_position().astype(np.float64)
                    target_orientation = arm.panda.get_orientation().astype(
                        np.float64
                    )  # [x,y,z,w]

                    drift = np.zeros(6, dtype=np.float64)
                    drift[:3] = target_position - pos_before
                    delta_quat = quat_multiply(
                        target_orientation, quat_inverse(ori_before)
                    )  # [x,y,z,w]
                    drift[3:] = quat_to_euler(delta_quat)  # 转为欧拉角

                    arm.panda.start_controller(ctrl)
                    last_gripper_cmd = gripper_cmd

                    if np.any(drift != 0):
                        delta = delta + drift

                if args.action_epsilon > 0.0:
                    small_mask = np.abs(delta) <= args.action_epsilon
                    if np.any(small_mask):
                        delta = delta.copy()
                        delta[small_mask] = 0.0

                if args.max_ee_step_m > 0.0:
                    step = delta[:3]
                    step_norm = float(np.linalg.norm(step))
                    if step_norm > float(args.max_ee_step_m):
                        delta = delta.copy()
                        delta[:3] = step / step_norm * float(args.max_ee_step_m)

                if np.any(delta != 0):
                    current_position = arm.panda.get_position().astype(np.float64)
                    target_position_new = target_position + delta[:3]

                    if args.max_position_error_m > 0.0:
                        pos_err = target_position_new - current_position
                        pos_err_norm = float(np.linalg.norm(pos_err))
                        if pos_err_norm > float(args.max_position_error_m):
                            target_position_new = (
                                current_position
                                + pos_err
                                / pos_err_norm
                                * float(args.max_position_error_m)
                            )

                    target_orientation_new = target_orientation
                    if np.any(delta[3:] != 0):
                        # 欧拉角增量转为四元数，使用 [x,y,z,w] 格式
                        delta_quat = euler_to_quat(
                            delta[3], delta[4], delta[5]
                        )  # [x,y,z,w]
                        target_orientation_new = quat_multiply(
                            target_orientation, delta_quat
                        )  # [x,y,z,w]

                    target_position = target_position_new
                    target_orientation = target_orientation_new
                    ctrl.set_control(
                        target_position, target_orientation
                    )  # 传入 [x,y,z,w] 格式

                if not enable_logging:
                    continue

                external_img = external_buf.get_latest()
                wrist_img = wrist_buf.get_latest()
                if external_img is None or wrist_img is None:
                    continue

                if external_img.shape != (args.image_hw, args.image_hw, 3):
                    continue
                if wrist_img.shape != (args.image_hw, args.image_hw, 3):
                    continue

                motion_norm = float(np.linalg.norm(delta))
                has_action = motion_norm >= motion_start_threshold or gripper_changed
                if has_action and not recording_started:
                    recording_started = True
                    recording_started_at = time.time()
                    print("[Recording] First action detected, start logging...")

                if recording_started:
                    state = arm.panda.get_state()
                    joint_pos = np.asarray(state.q, dtype=np.float32)
                    joint_vel = np.asarray(state.dq, dtype=np.float32)

                    actions = np.concatenate(
                        [joint_vel, [np.float32(gripper_state)]], dtype=np.float32
                    )
                    gripper_pos = np.asarray(
                        [np.float32(gripper_state)], dtype=np.float32
                    )
                    blank = np.zeros_like(external_img)

                    dataset.add_frame(
                        {
                            "exterior_image_1_left": external_img,
                            "exterior_image_2_left": blank,
                            "wrist_image_left": wrist_img,
                            "joint_position": joint_pos,
                            "gripper_position": gripper_pos,
                            "actions": actions,
                            "task": args.instruction,
                        }
                    )
                    frame_count += 1
                    if frame_count % 50 == 0:
                        print(f"[Recording] {frame_count} frames", end="\r")

    except RuntimeError as exc:
        msg = str(exc)
        if "motion aborted by reflex" in msg or "joint_velocity_violation" in msg:
            reflex_error_occurred = True
            print("\n[Error] Franka reflex triggered; aborting teleop safely.")
            print(f"[Error] {msg}")
            print(
                "[Hint] Try smaller --translation-speed/--rotation-speed, or increase --control-frequency. "
                "The new default --speed-units=per_second is recommended."
            )
        else:
            raise
    except KeyboardInterrupt:
        print("\n[Recording] Ctrl+C detected, stopping...")
    finally:
        try:
            arm.panda.stop_controller()
        except Exception:
            pass

        external_stop.set()
        wrist_stop.set()
        external_thread.join(timeout=2.0)
        wrist_thread.join(timeout=2.0)

        try:
            external_cam.close()
        except Exception:
            pass
        try:
            wrist_cam.close()
        except Exception:
            pass

        try:
            teleop.disconnect()
        except Exception:
            pass

        try:
            arm.cleanup()
        except Exception:
            pass

        if enable_logging and dataset is not None and not reflex_error_occurred:
            try:
                if frame_count > 0:
                    try:
                        _prepare_episode_for_save(dataset)
                        dataset.save_episode()
                        print(f"\n[Recording] Saved episode with {frame_count} frames")
                    except Exception as exc:
                        print(f"\n[Error] Failed to save last episode: {exc}")
                        try:
                            dataset.clear_episode_buffer()
                        except Exception:
                            pass
            finally:
                try:
                    dataset.stop_image_writer()
                except Exception:
                    pass
        elif enable_logging and dataset is not None and reflex_error_occurred:
            try:
                dataset.clear_episode_buffer()
            except Exception:
                pass
            try:
                dataset.stop_image_writer()
            except Exception:
                pass


if __name__ == "__main__":
    main()
