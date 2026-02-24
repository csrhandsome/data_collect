#!/usr/bin/env python3
"""
Collect Franka teleop data into a LeRobot dataset using DROID-style keys.

- Cameras: external + wrist (2x RealSense)
- Control: joint velocity via IntegratedVelocity + DROID dm_control IK solver
- Actions: normalized joint velocity [-1, 1] + gripper state
- Output: LeRobot dataset (no intermediate HDF5)

uv run xbox_openpi_lerobot_franka.py \
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
from droid_ik_solver import DroidIKSolver
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
        "--sensitivity",
        type=float,
        default=1.0,
        help=(
            "Joystick sensitivity multiplier (0-1]. "
            "1.0 = full DROID speed (0.075 m/step translation, 0.15 rad/step rotation). "
            "0.5 = half speed. Applied before IK solver."
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
    print(f"Sensitivity: {args.sensitivity} (1.0 = full DROID speed)")
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
        date = "2_22"
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

    # --- DROID IK solver (replaces simple Jacobian pseudoinverse) ---
    print("Initializing DROID IK solver (dm_control)...")
    ik_solver = DroidIKSolver(control_hz=float(args.control_frequency))

    from panda_py import controllers

    ctrl = controllers.IntegratedVelocity()
    arm.panda.start_controller(ctrl)
    time.sleep(0.5)
    ctrl.set_control(np.zeros(7))

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
                        ctrl.set_control(np.zeros(7))  # 停止运动
                        try:
                            _prepare_episode_for_save(dataset)
                            dataset.save_episode()
                            print(
                                f"[Recording] Saved episode with {frame_count} frames"
                            )
                        except Exception as exc:
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

                        ctrl = controllers.IntegratedVelocity()
                        arm.panda.start_controller(ctrl)
                        ctrl.set_control(np.zeros(7))
                        continue

                    print("\n[Control] X/O pressed, stopping...")
                    break

                # --- Cartesian velocity [-1, 1] (DROID semantics) ---
                # Joystick input [-1, 1] * sensitivity → cartesian_velocity for IK solver
                cart_vel = np.zeros(6, dtype=np.float64)
                sens = float(args.sensitivity)

                cart_vel[0] = float(action.get("delta_x", 0.0)) * sens
                cart_vel[1] = float(action.get("delta_y", 0.0)) * sens
                cart_vel[2] = (axes["rt"] - axes["lt"]) * sens
                if getattr(teleop, "joystick", None) is None:
                    cart_vel[2] = float(action.get("delta_z", 0.0)) * sens

                if buttons["l1"]:
                    cart_vel[3] -= sens
                if buttons["r1"]:
                    cart_vel[3] += sens
                cart_vel[4] = (-axes["right_y"]) * sens
                cart_vel[5] = (axes["right_x"]) * sens

                if args.action_epsilon > 0.0:
                    small_mask = np.abs(cart_vel) <= args.action_epsilon
                    if np.any(small_mask):
                        cart_vel[small_mask] = 0.0

                # --- DROID IK solver: cart_vel [-1,1] → joint_delta + normalized_action ---
                robot_state = arm.panda.get_state()
                qpos = np.asarray(robot_state.q, dtype=np.float64)
                qvel = np.asarray(robot_state.dq, dtype=np.float64)
                joint_delta, normalized_action = ik_solver.solve(cart_vel, qpos, qvel)

                # Apply position increment (same as inference)
                ctrl.set_control(joint_delta)

                # --- Gripper ---
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
                    ctrl.set_control(np.zeros(7))
                    arm.panda.stop_controller()
                    if gripper_cmd > 0.5:
                        arm.gripper_open()
                        gripper_state = 1.0
                    else:
                        arm.gripper_close()
                        gripper_state = 0.0
                    ctrl = controllers.IntegratedVelocity()
                    arm.panda.start_controller(ctrl)
                    ctrl.set_control(np.zeros(7))
                    last_gripper_cmd = gripper_cmd

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

                motion_norm = float(np.linalg.norm(cart_vel))
                has_action = motion_norm >= motion_start_threshold or gripper_changed
                if has_action and not recording_started:
                    recording_started = True
                    recording_started_at = time.time()
                    print("[Recording] First action detected, start logging...")

                if recording_started:
                    joint_pos = np.asarray(robot_state.q, dtype=np.float32)

                    actions = np.concatenate(
                        [
                            normalized_action.astype(np.float32),
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
            print("[Hint] Try smaller --sensitivity, or increase --control-frequency.")
        else:
            raise
    except KeyboardInterrupt:
        print("\n[Recording] Ctrl+C detected, stopping...")
    finally:
        try:
            ctrl.set_control(np.zeros(7))
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
