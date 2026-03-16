#!/usr/bin/env python3
"""
Collect Franka teleop data into a LeRobot dataset using VR (teleop_xr) input.

- Cameras: external + wrist (2x RealSense)
- Control: VR end-effector pose -> IK -> joint position controller
- Actions: absolute joint target + gripper state
- Output: LeRobot dataset (no intermediate HDF5)

Deadman: hold both VR triggers (long press) to enable arm movement.
X (left controller): switch to the second prompt.
Recording: start automatically when motion begins.
Y (left controller): save current episode and return to the start pose.
A (right controller): gripper close.  B: gripper open.

uv run vr_openpi_lerobot_joint_force.py \
  --instruction "Pick up the wide-mouth bottle" \
  --second-instruction "Pick up the narrow-mouth bottle" \
  --external-camera-serial 825412070292 \
  --wrist-camera-serial 825412070487 \
  --color-only
"""

import argparse
import time
from pathlib import Path
from typing import Optional
import sys
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from control.img_util import center_crop_and_resize_rgb_uint8
from control.soft_gripper_control import DH5Gripper
from control.vr_input import VRInputProcess
from control.vr_input_mapper import VREEPoseMapper
from control.dual_camera_manager import DualRealsenseManager
from ik_solver import FrankaJointIKSolver
from control.robotic_arm_controller import RoboticArmControler


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
            "gripper_image_left": {
                "dtype": "image",
                "shape": (image_hw, image_hw, 3),
                "names": ["height", "width", "channel"],
            },
            "gripper_image_right": {
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
        image_writer_processes=0,
    )


def _load_or_create_dataset(
    repo_id: str, *, fps: float, image_hw: int, root: Path
) -> LeRobotDataset:
    def looks_like_lerobot_dataset(path: Path) -> bool:
        return (path / "meta" / "info.json").is_file() and (
            path / "meta" / "episodes.jsonl"
        ).is_file()

    def resume_existing_dataset_for_recording(path: Path) -> LeRobotDataset:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.common.datasets.video_utils import get_safe_default_codec

        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=path)

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
                "Fix the dataset first, then retry."
            )

        next_ep = meta.total_episodes
        images_dir = meta.root / "images"
        if images_dir.is_dir():
            leftover = list(images_dir.rglob(f"episode_{next_ep:06d}"))
            if leftover:
                raise RuntimeError(
                    "Cannot resume: found leftover temporary images for the next episode. "
                    f"Please remove '{images_dir}' (or the episode_{next_ep:06d} folder) and retry."
                )

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
                f".venv/bin/python data_analysis/delete_latest_episode.py --dataset {root}"
            ) from exc

    return _create_dataset(repo_id, fps=fps, image_hw=image_hw, root=root)


def _prepare_episode_for_save(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        return
    gripper_values = dataset.episode_buffer.get("gripper_position")
    if not isinstance(gripper_values, list) or not gripper_values:
        return
    dataset.episode_buffer["gripper_position"] = [
        float(v.reshape(-1)[0]) if isinstance(v, np.ndarray) else float(v)
        for v in gripper_values
    ]


def _wait_for_soft_gripper_frames(
    soft_gripper: DH5Gripper, *, timeout_s: float
) -> tuple[np.ndarray, np.ndarray]:
    start = time.time()
    while True:
        right_img, left_img = soft_gripper.dual_camera_manager.get_images()
        if left_img is not None and right_img is not None:
            return left_img, right_img
        if timeout_s > 0 and (time.time() - start) > timeout_s:
            raise RuntimeError(
                f"Soft gripper camera timeout after {timeout_s:.1f} seconds."
            )
        time.sleep(0.01)


def _get_soft_gripper_images(
    soft_gripper: DH5Gripper, *, crop_scale: float, image_hw: int
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    right_img, left_img = soft_gripper.dual_camera_manager.get_images()
    if left_img is None or right_img is None:
        return None, None
    left_rgb = center_crop_and_resize_rgb_uint8(
        left_img,
        crop_scale=crop_scale,
        out_hw=image_hw,
    )
    right_rgb = center_crop_and_resize_rgb_uint8(
        right_img,
        crop_scale=crop_scale,
        out_hw=image_hw,
    )
    return left_rgb, right_rgb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Franka data into LeRobot via VR (teleop_xr)"
    )
    parser.add_argument("--repo-id", type=str, default="openpi/franka_franka_lerobot")
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--second-instruction", type=str, default="")
    parser.add_argument("--control-frequency", type=float, default=30.0)
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.9,
        help="Global multiplier on the per-step EE command increments.",
    )
    parser.add_argument(
        "--max-ee-translation-step",
        type=float,
        default=0.04,  # 0.02
        help="Maximum EE translation increment per control cycle (m/step) at full VR input.",
    )
    parser.add_argument(
        "--max-ee-rotation-step",
        type=float,
        default=0.04,
        help="Maximum EE rotation increment per control cycle (rad/step) at full VR input.",
    )
    parser.add_argument("--action-epsilon", type=float, default=1e-6)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--color-only", action="store_true")
    parser.add_argument("--camera-startup-timeout-s", type=float, default=10.0)
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument("--max-duration", type=float, default=3600.0)
    parser.add_argument("--external-camera-serial", type=str, default=None)
    parser.add_argument("--wrist-camera-serial", type=str, default=None)
    parser.add_argument(
        "--soft-gripper-port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port for the DH soft gripper controller.",
    )
    parser.add_argument(
        "--gripper-right-camera-device",
        type=str,
        default="1",
        help="OpenCV device index/path for the right soft-gripper camera.",
    )
    parser.add_argument(
        "--gripper-left-camera-device",
        type=str,
        default="2",
        help="OpenCV device index/path for the left soft-gripper camera.",
    )
    parser.add_argument(
        "--soft-gripper-force",
        type=int,
        default=50,
        help="DH soft-gripper force in [20, 100].",
    )
    parser.add_argument(
        "--soft-gripper-velocity",
        type=int,
        default=100,
        help="DH soft-gripper velocity in [0, 1000].",
    )
    parser.add_argument(
        "--soft-gripper-axis-threshold",
        type=float,
        default=0.55,
        help="Thumbstick threshold used by `DH5Gripper.update_from_vr()`.",
    )
    parser.add_argument(
        "--soft-gripper-step-interval",
        type=float,
        default=0.08,
        help="Minimum interval between adjacent soft-gripper level steps.",
    )
    parser.add_argument("--image-hw", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, default=0.9)
    parser.add_argument("--no-logging", action="store_true")
    # VR-specific
    parser.add_argument("--vr-host", type=str, default="0.0.0.0")
    parser.add_argument("--vr-port", type=int, default=4443)
    parser.add_argument(
        "--vr-long-press-s",
        type=float,
        default=0.5,
        help="Seconds both triggers must be held to enable arm movement.",
    )
    parser.add_argument(
        "--vr-translation-scale",
        type=float,
        default=0.04,
        help="VR translation delta (m) between adjacent samples that maps to a full EE translation step. Smaller values make motion faster.",
    )
    parser.add_argument(
        "--vr-rotation-scale",
        type=float,
        default=0.20,
        help="VR rotation delta (rad) between adjacent samples that maps to a full EE rotation step. Smaller values make rotation faster.",
    )
    parser.add_argument(
        "--vr-position-alpha",
        type=float,
        default=0.6,
        help="Right-controller position interpolation factor in (0, 1]. Smaller values are smoother but add latency; 1 disables input smoothing.",
    )
    parser.add_argument(
        "--vr-rotation-alpha",
        type=float,
        default=0.35,
        help="Right-controller rotation interpolation factor in (0, 1]. Smaller values are smoother but add latency; 1 disables input smoothing.",
    )
    parser.add_argument(
        "--max-ee-translation",
        type=float,
        default=0.5,
        help="Optional workspace half-range around the engagement pose (m). Set 0 to disable the translation clamp.",
    )
    parser.add_argument(
        "--max-ee-rotation",
        type=float,
        default=1.2,
        help="Optional rotational half-range around the engagement pose (rad). Set 0 to disable the rotation clamp.",
    )
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.0,
        help="Optional hard clip for per-cycle joint position change (rad). Set 0 to disable clipping.",
    )
    parser.add_argument(
        "--joint-velocity-limit",
        type=float,
        default=0.45,
        help="Clip joint velocity feedforward (rad/s). Set 0 to disable feedforward.",
    )
    parser.add_argument(
        "--vr-enable-rotation",
        action="store_true",
        help="Enable rotation control from VR wrist. Off by default to avoid IK chaos.",
    )

    args = parser.parse_args()
    sys.setswitchinterval(0.0005)
    if args.control_frequency <= 0:
        raise ValueError("--control-frequency must be > 0")
    if args.vr_translation_scale <= 0:
        raise ValueError("--vr-translation-scale must be > 0")
    if args.vr_rotation_scale <= 0:
        raise ValueError("--vr-rotation-scale must be > 0")
    if not 0.0 < args.vr_position_alpha <= 1.0:
        raise ValueError("--vr-position-alpha must be in (0, 1]")
    if not 0.0 < args.vr_rotation_alpha <= 1.0:
        raise ValueError("--vr-rotation-alpha must be in (0, 1]")
    if args.max_ee_translation_step <= 0:
        raise ValueError("--max-ee-translation-step must be > 0")
    if args.max_ee_rotation_step <= 0:
        raise ValueError("--max-ee-rotation-step must be > 0")
    if args.max_ee_translation < 0:
        raise ValueError("--max-ee-translation must be >= 0")
    if args.max_ee_rotation < 0:
        raise ValueError("--max-ee-rotation must be >= 0")
    if args.max_joint_delta < 0:
        raise ValueError("--max-joint-delta must be >= 0")
    if args.joint_velocity_limit < 0:
        raise ValueError("--joint-velocity-limit must be >= 0")
    if not 20 <= args.soft_gripper_force <= 100:
        raise ValueError("--soft-gripper-force must be in [20, 100]")
    if not 0 <= args.soft_gripper_velocity <= 1000:
        raise ValueError("--soft-gripper-velocity must be in [0, 1000]")
    if not 0.0 < args.soft_gripper_axis_threshold <= 1.0:
        raise ValueError("--soft-gripper-axis-threshold must be in (0, 1]")
    if args.soft_gripper_step_interval <= 0.0:
        raise ValueError("--soft-gripper-step-interval must be > 0")

    def _parse_camera_device(device: str) -> int | str:
        return int(device) if str(device).lstrip("-").isdigit() else device

    enable_logging = not args.no_logging
    if enable_logging and not args.instruction.strip():
        raise ValueError("--instruction is required when logging is enabled")

    print("=" * 70)
    print("Franka LeRobot data collection (VR teleop)")
    print("=" * 70)
    dq_clip_str = (
        "off" if args.max_joint_delta <= 0 else f"{args.max_joint_delta:.3f}rad/step"
    )
    qvel_ff_str = (
        "off"
        if args.joint_velocity_limit <= 0
        else f"{args.joint_velocity_limit:.2f}rad/s"
    )
    trans_limit_str = (
        "off" if args.max_ee_translation <= 0 else f"±{args.max_ee_translation:.3f}m"
    )
    rot_limit_str = (
        "off" if args.max_ee_rotation <= 0 else f"±{args.max_ee_rotation:.3f}rad"
    )
    print(f"Control frequency: {args.control_frequency} Hz")
    print(f"Sensitivity: {args.sensitivity}")
    print(
        "Joint control tuning: "
        f"step_xyz={args.max_ee_translation_step:.3f}m, "
        f"step_rot={args.max_ee_rotation_step:.3f}rad, "
        f"limit_xyz={trans_limit_str}, "
        f"limit_rot={rot_limit_str}, "
        f"vr_rot={'on' if args.vr_enable_rotation else 'off'}, "
        f"dq_clip={dq_clip_str}, "
        f"qdot_ff={qvel_ff_str}"
    )
    print(
        "VR input smoothing: "
        f"pos_alpha={args.vr_position_alpha:.2f}, "
        f"rot_alpha={args.vr_rotation_alpha:.2f}"
    )
    print(
        f"Camera stream: {args.camera_width}x{args.camera_height}@{args.camera_fps} "
        f"(depth={'off' if args.color_only else 'on'})"
    )
    print(f"VR server: {args.vr_host}:{args.vr_port}")
    print(f"External camera serial: {args.external_camera_serial}")
    print(f"Wrist camera serial: {args.wrist_camera_serial}")
    print(
        "Soft gripper: "
        f"port={args.soft_gripper_port}, "
        f"left_cam={args.gripper_left_camera_device}, "
        f"right_cam={args.gripper_right_camera_device}, "
        f"force={args.soft_gripper_force}, "
        f"velocity={args.soft_gripper_velocity}, "
        f"axis_threshold={args.soft_gripper_axis_threshold:.2f}, "
        f"step_interval={args.soft_gripper_step_interval:.2f}s"
    )
    if enable_logging:
        date = "3_8"
        args.repo_id = f"{args.repo_id}_{date}"
        print(f"Logging: enabled (max {args.max_duration} s)")
        print(f"Instruction: {args.instruction}")
        if args.second_instruction.strip():
            print(f"Second instruction: {args.second_instruction}")
        print(f"LeRobot repo_id: {args.repo_id}")
    else:
        print("Logging: disabled")
    print("=" * 70)

    # --- VR input (separate process — immune to main-process GIL) ---
    print("Starting VR input reader (separate process)...")
    vr_reader = VRInputProcess(
        host=args.vr_host,
        port=args.vr_port,
        long_press_s=args.vr_long_press_s,
        position_alpha=args.vr_position_alpha,
        rotation_alpha=args.vr_rotation_alpha,
    )
    vr_reader.start()

    max_ee_translation = float(args.max_ee_translation)
    max_ee_rotation = float(args.max_ee_rotation)
    max_ee_translation_step = float(args.max_ee_translation_step)
    max_ee_rotation_step = float(args.max_ee_rotation_step)
    max_joint_delta = float(args.max_joint_delta)

    vr_mapper = VREEPoseMapper(
        translation_scale=args.vr_translation_scale,
        rotation_scale=args.vr_rotation_scale,
        max_translation_step=max_ee_translation_step,
        max_rotation_step=max_ee_rotation_step,
        translation_limit=max_ee_translation,
        rotation_limit=max_ee_rotation,
        sensitivity=args.sensitivity,
    )

    # --- Franka arm ---
    print("Initializing Franka arm...")
    arm = RoboticArmControler()

    print("Initializing DH soft gripper...")
    soft_gripper = DH5Gripper(
        com=args.soft_gripper_port,
        right_camera_device=_parse_camera_device(args.gripper_right_camera_device),
        left_camera_device=_parse_camera_device(args.gripper_left_camera_device),
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        camera_timeout_ms=args.camera_timeout_ms,
        vr_axis_threshold=float(args.soft_gripper_axis_threshold),
        vr_step_interval_s=float(args.soft_gripper_step_interval),
    )
    soft_gripper.set_force(args.soft_gripper_force)
    soft_gripper.set_velocity(args.soft_gripper_velocity)

    # --- Cameras ---
    print("Initializing RealSense cameras...")
    camera_manager = DualRealsenseManager(
        external_serial=args.external_camera_serial,
        wrist_serial=args.wrist_camera_serial,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        enable_depth=not args.color_only,
        background_poll=True,
        background_timeout_ms=int(args.camera_timeout_ms),
        crop_scale=float(args.crop_scale),
        out_hw=int(args.image_hw),
    )
    camera_manager.connect()

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

    print("[Camera] Waiting for first frames...")
    camera_manager.wait_for_frames(timeout_s=float(args.camera_startup_timeout_s))
    print("[Camera] First frames acquired, ready to record!")

    print("[SoftGripper] Waiting for first gripper-camera frames...")
    _wait_for_soft_gripper_frames(
        soft_gripper, timeout_s=float(args.camera_startup_timeout_s)
    )
    print("[SoftGripper] First gripper-camera frames acquired, ready to record!")

    print("Opening gripper...")
    soft_gripper.set_position(0, wait=True)
    print("Moving to start position...")
    arm.move_to_start()
    if not arm.wait_until_stopped():
        max_vel = float(np.max(np.abs(np.asarray(arm.panda.get_state().dq))))
        print(
            "[Warning] Robot did not fully stop after move_to_start: "
            f"max_vel={max_vel:.4f} rad/s"
        )

    # --- EE pose IK solver ---
    print("Initializing Franka EE pose IK solver (dm_control)...")
    ik_solver = FrankaJointIKSolver(
        linear_tol=2e-3,
        angular_tol=5e-3,
        max_steps=50,
        num_attempts=1,
    )
    joint_limits = ik_solver.joint_limits

    from panda_py import controllers

    def _current_qpos() -> np.ndarray:
        robot_state = arm.panda.get_state()
        return np.asarray(robot_state.q, dtype=np.float64)

    def _hold_current_joint_position(controller) -> None:
        qpos = _current_qpos()
        controller.set_control(qpos, np.zeros(7, dtype=np.float64))

    def _start_joint_position_controller(settle_s: float = 0.0):
        controller = controllers.JointPosition()
        arm.panda.start_controller(controller)
        if settle_s > 0.0:
            time.sleep(settle_s)
        _hold_current_joint_position(controller)
        return controller

    ctrl = _start_joint_position_controller(settle_s=0.5)

    active_instruction = args.instruction
    gripper_state = float(soft_gripper.gripper_open_ratio.reshape(-1)[0])

    recording_started = False
    recording_started_at: Optional[float] = None
    frame_count = 0
    motion_start_threshold = max(float(args.action_epsilon), 1e-3)
    reflex_error_occurred = False
    prev_x_pressed = False
    prev_y_pressed = False
    last_ik_warning_at = 0.0

    def _reset_robot_to_start() -> None:
        nonlocal ctrl, gripper_state

        _hold_current_joint_position(ctrl)
        arm.panda.stop_controller()
        try:
            soft_gripper.set_gripper_level(0)
        except Exception as exc:
            print(f"[Warning] Failed to open soft gripper during reset: {exc}")
        gripper_state = float(soft_gripper.gripper_open_ratio.reshape(-1)[0])
        print("[Control] Moving to start position...")
        arm.move_to_start()
        if not arm.wait_until_stopped():
            max_vel = float(np.max(np.abs(np.asarray(arm.panda.get_state().dq))))
            print(
                "[Warning] Robot did not fully stop after move_to_start: "
                f"max_vel={max_vel:.4f} rad/s"
            )
        ctrl = _start_joint_position_controller(settle_s=0.0)
        vr_mapper.reset()

    def _finish_episode(*, save_episode: bool, message: str) -> None:
        nonlocal frame_count, recording_started, recording_started_at

        print(message)
        if dataset is not None:
            if save_episode and frame_count > 0:
                try:
                    _prepare_episode_for_save(dataset)
                    dataset.save_episode()
                    print(f"[Recording] Saved episode with {frame_count} frames")
                except Exception as exc:
                    print(f"[Error] Failed to save episode: {exc}")
                    try:
                        dataset.clear_episode_buffer()
                    except Exception:
                        pass
            else:
                try:
                    dataset.clear_episode_buffer()
                except Exception:
                    pass

        frame_count = 0
        recording_started = False
        recording_started_at = None
        _reset_robot_to_start()

    print("\nControl mapping (VR):")
    print("  Hold both triggers (long press): enable arm movement")
    print(
        "  Right controller pose: snapshot-based EE pose control -> IK -> joint position"
    )
    print("  Release / re-hold triggers: re-anchor the VR neutral pose")
    print("  Right stick Y: step soft gripper close/open")
    print("  A (right): gripper fully close | B (right): gripper fully open")
    print("  X (left): switch to second prompt")
    print("  Recording: starts automatically when motion begins")
    print("  Y (left): save current episode and return to start")

    try:
        with arm.panda.create_context(frequency=args.control_frequency) as ctx:
            while ctx.ok():
                if (
                    enable_logging
                    and recording_started
                    and recording_started_at is not None
                    and (time.time() - recording_started_at) > args.max_duration
                ):
                    _finish_episode(
                        save_episode=frame_count > 0,
                        message="\n[Recording] Max duration reached, saving episode and returning to start...",
                    )
                    continue

                vr = vr_reader.latest

                x_pressed = bool(getattr(vr, "x_pressed", False))
                y_pressed = bool(getattr(vr, "y_pressed", False))
                x_edge = x_pressed and not prev_x_pressed
                y_edge = y_pressed and not prev_y_pressed
                prev_x_pressed = x_pressed
                prev_y_pressed = y_pressed

                if x_edge:
                    if not enable_logging:
                        print("\n[Prompt] Ignored X press because logging is disabled.")
                    elif not args.second_instruction.strip():
                        print(
                            "\n[Prompt] X pressed, but --second-instruction is empty; keeping current prompt."
                        )
                    elif active_instruction == args.second_instruction:
                        print("\n[Prompt] Second prompt is already active.")
                    else:
                        active_instruction = args.second_instruction
                        print(
                            f"\n[Prompt] Switched active prompt to second prompt: {active_instruction}"
                        )
                    continue

                if y_edge:
                    if not enable_logging:
                        print(
                            "\n[Recording] Ignored Y press because logging is disabled."
                        )
                    else:
                        _finish_episode(
                            save_episode=frame_count > 0,
                            message=(
                                "\n[Control] Y pressed, saving episode and returning to start..."
                                if frame_count > 0
                                else "\n[Control] Y pressed, resetting with no captured frames."
                            ),
                        )
                    continue

                robot_state = arm.panda.get_state()
                qpos = np.asarray(robot_state.q, dtype=np.float64)
                ee_pos, ee_quat = ik_solver.forward_kinematics(qpos)
                target_ee_pos, target_ee_quat = vr_mapper.map(vr, ee_pos, ee_quat)

                joint_delta = np.zeros(7, dtype=np.float64)
                normalized_action = np.zeros(7, dtype=np.float64)
                qpos_cmd = qpos.copy()
                qvel_cmd = np.zeros(7, dtype=np.float64)

                if vr.arm_enabled:
                    target_qpos = ik_solver.solve_pose(
                        target_ee_pos,
                        target_ee_quat,
                        initial_joint_configuration=qpos,
                        nullspace_reference=qpos,
                        early_stop=True,
                        num_attempts=1,
                        stop_on_first_successful_attempt=True,
                    )
                    if target_qpos is None:
                        now = time.time()
                        if now - last_ik_warning_at > 1.0:
                            print(
                                "[Warning] IK failed for current EE target; holding position."
                            )
                            last_ik_warning_at = now
                    else:
                        target_qpos = np.clip(
                            np.asarray(target_qpos, dtype=np.float64),
                            joint_limits[:, 0],
                            joint_limits[:, 1],
                        )
                        joint_delta = target_qpos - qpos
                        if max_joint_delta > 0.0:
                            joint_delta = np.clip(
                                joint_delta,
                                -max_joint_delta,
                                max_joint_delta,
                            )
                            normalized_action = joint_delta / max_joint_delta
                        else:
                            normalized_action = joint_delta.copy()

                if args.action_epsilon > 0.0:
                    small_mask = np.abs(joint_delta) <= args.action_epsilon
                    if np.any(small_mask):
                        normalized_action[small_mask] = 0.0
                        joint_delta[small_mask] = 0.0

                qpos_cmd = np.clip(
                    qpos + joint_delta,
                    joint_limits[:, 0],
                    joint_limits[:, 1],
                )
                if args.joint_velocity_limit > 0.0:
                    qvel_cmd = np.clip(
                        joint_delta * float(args.control_frequency),
                        -float(args.joint_velocity_limit),
                        float(args.joint_velocity_limit),
                    )
                else:
                    qvel_cmd = np.zeros(7, dtype=np.float64)
                ctrl.set_control(qpos_cmd, qvel_cmd)

                # --- Gripper ---
                gripper_changed = False
                if vr.gripper_close:
                    gripper_changed = soft_gripper.set_gripper_level(
                        soft_gripper.max_gripper_level,
                        wait=False,
                    )
                elif vr.gripper_open:
                    gripper_changed = soft_gripper.set_gripper_level(0, wait=False)
                else:
                    gripper_changed = soft_gripper.update_from_vr(vr, wait=False)

                gripper_state = float(soft_gripper.gripper_open_ratio.reshape(-1)[0])

                if not enable_logging:
                    continue

                external_img, wrist_img = camera_manager.get_images()
                gripper_left_img, gripper_right_img = _get_soft_gripper_images(
                    soft_gripper,
                    crop_scale=float(args.crop_scale),
                    image_hw=int(args.image_hw),
                )
                if external_img is None or wrist_img is None:
                    continue
                if gripper_left_img is None or gripper_right_img is None:
                    continue

                if external_img.shape != (args.image_hw, args.image_hw, 3):
                    continue
                if wrist_img.shape != (args.image_hw, args.image_hw, 3):
                    continue
                if gripper_left_img.shape != (args.image_hw, args.image_hw, 3):
                    continue
                if gripper_right_img.shape != (args.image_hw, args.image_hw, 3):
                    continue

                motion_norm = float(np.linalg.norm(joint_delta))
                has_action = motion_norm >= motion_start_threshold or gripper_changed

                if has_action and not recording_started:
                    recording_started = True
                    recording_started_at = time.time()
                    print(
                        f"\n[Recording] First motion detected, start logging with prompt: {active_instruction}"
                    )

                if recording_started:
                    joint_pos = np.asarray(robot_state.q, dtype=np.float32)

                    actions = np.concatenate(
                        [
                            qpos_cmd.astype(np.float32),
                            [np.float32(gripper_state)],
                        ],
                        dtype=np.float32,
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
                            "gripper_image_left": gripper_left_img,
                            "gripper_image_right": gripper_right_img,
                            "joint_position": joint_pos,
                            "gripper_position": gripper_pos,
                            "actions": actions,
                            "task": active_instruction,
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
                "[Hint] Try smaller --sensitivity, --max-ee-translation-step, "
                "or --max-ee-rotation-step."
            )
        else:
            raise
    except KeyboardInterrupt:
        print("\n[Recording] Ctrl+C detected, stopping...")
    finally:
        try:
            _hold_current_joint_position(ctrl)
            arm.panda.stop_controller()
        except Exception:
            pass

        try:
            camera_manager.close()
        except Exception:
            pass

        try:
            soft_gripper.close()
        except Exception:
            pass

        try:
            vr_reader.stop()
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
