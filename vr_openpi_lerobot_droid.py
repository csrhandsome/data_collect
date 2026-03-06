#!/usr/bin/env python3
"""
Collect Franka teleop data into a LeRobot dataset using VR (teleop_xr) input.

- Cameras: external + wrist (2x RealSense)
- Control: joint velocity via IntegratedVelocity + DROID dm_control IK solver
- Actions: normalized joint velocity [-1, 1] + gripper state
- Output: LeRobot dataset (no intermediate HDF5)

Deadman: hold both VR triggers (long press) to enable arm movement.
X / Y (left controller): save episode + reset.
A (right controller): gripper close.  B: gripper open.

uv run vr_lerobot_franka.py \
  --instruction "Pick up the wide-mouth bottle" \
  --external-camera-serial 825412070292 \
  --wrist-camera-serial 825412070487 \
  --color-only
"""

import argparse
import threading
import time
from pathlib import Path
from typing import Optional
import sys
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from control.vr_input import VRInputProcess
from control.vr_input_mapper import VRInputMapper
from realsense_connector import RealSenseConnector
from ik_solver import DroidIKSolver
from control.robotic_arm_controller import RoboticArmControler
from control.robotic_arm_controller import _camera_capture_worker
from control.robotic_arm_controller import _LatestFrameBuffer


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
                f".venv/bin/python data_analysis/delete_latest_episode.py --dataset {root} --apply"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Franka data into LeRobot via VR (teleop_xr)"
    )
    parser.add_argument("--repo-id", type=str, default="openpi/franka_droid_lerobot")
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--control-frequency", type=float, default=15.0)
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="Cartesian velocity multiplier (0-1]. 1.0 = full DROID speed.",
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
        default=0.05,
        help="VR displacement (m) from neutral that maps to full speed. 0.05 = 5cm.",
    )
    parser.add_argument(
        "--vr-rotation-scale",
        type=float,
        default=0.8,
        help="VR rotation (rad) from neutral that maps to full speed. 0.35 ≈ 20 deg.",
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

    enable_logging = not args.no_logging
    if enable_logging and not args.instruction.strip():
        raise ValueError("--instruction is required when logging is enabled")

    print("=" * 70)
    print("Franka LeRobot data collection (VR teleop)")
    print("=" * 70)
    print(f"Control frequency: {args.control_frequency} Hz")
    print(f"Sensitivity: {args.sensitivity}")
    print(
        f"Camera stream: {args.camera_width}x{args.camera_height}@{args.camera_fps} "
        f"(depth={'off' if args.color_only else 'on'})"
    )
    print(f"VR server: {args.vr_host}:{args.vr_port}")
    print(f"External camera serial: {args.external_camera_serial}")
    print(f"Wrist camera serial: {args.wrist_camera_serial}")
    if enable_logging:
        date = "3_8"
        args.repo_id = f"{args.repo_id}_{date}"
        print(f"Logging: enabled (max {args.max_duration} s)")
        print(f"Instruction: {args.instruction}")
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
    )
    vr_reader.start()

    vr_mapper = VRInputMapper(
        translation_scale=args.vr_translation_scale,
        rotation_scale=args.vr_rotation_scale,
    )

    # --- Franka arm ---
    print("Initializing Franka arm...")
    arm = RoboticArmControler()

    # --- Cameras ---
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
                f"Camera timeout: failed to get first frames within {args.camera_startup_timeout_s:.1f}s."
            )
        time.sleep(0.01)
    print("[Camera] First frames acquired, ready to record!")

    print("Opening gripper...")
    arm.gripper_open()
    print("Moving to start position...")
    arm.move_to_start()

    # --- DROID IK solver ---
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
    gripper_busy = False
    gripper_switch_cooldown_s = 0.12
    reflex_error_occurred = False
    prev_save_pressed = False

    print("\nControl mapping (VR):")
    print("  Hold both triggers (long press): enable arm movement")
    print("  Right controller pose: 6DoF arm control")
    print("  A (right): gripper close | B (right): gripper open")
    print("  X/Y (left): save episode + reset (press again with no frames to quit)")

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

                vr = vr_reader.latest

                # --- Save / Quit (edge: rising) ---
                save_pressed = vr.save_pressed
                save_edge = save_pressed and not prev_save_pressed
                prev_save_pressed = save_pressed

                if save_edge:
                    if enable_logging and frame_count > 0:
                        print(
                            "\n[Control] X/Y pressed, saving episode and resetting..."
                        )
                        ctrl.set_control(np.zeros(7))
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
                        vr_mapper.reset()
                        continue

                    print("\n[Control] X/Y pressed, stopping...")
                    break

                # --- Cartesian velocity from VR ---
                cart_vel = vr_mapper.map(vr) * float(args.sensitivity)

                if args.action_epsilon > 0.0:
                    small_mask = np.abs(cart_vel) <= args.action_epsilon
                    if np.any(small_mask):
                        cart_vel[small_mask] = 0.0

                # --- DROID IK: cart_vel [-1,1] -> joint_delta + normalized_action ---
                robot_state = arm.panda.get_state()
                qpos = np.asarray(robot_state.q, dtype=np.float64)
                qvel = np.asarray(robot_state.dq, dtype=np.float64)
                joint_delta, normalized_action = ik_solver.solve(cart_vel, qpos, qvel)

                if not gripper_busy:
                    ctrl.set_control(joint_delta)

                # --- Gripper ---
                gripper_changed = False
                gripper_cmd = last_gripper_cmd
                if vr.gripper_close:
                    gripper_cmd = 0.0
                elif vr.gripper_open:
                    gripper_cmd = 1.0

                now = time.time()
                if now - last_gripper_switch_time < gripper_switch_cooldown_s:
                    gripper_cmd = last_gripper_cmd

                if gripper_cmd != last_gripper_cmd and not gripper_busy:
                    gripper_changed = True
                    last_gripper_switch_time = now
                    gripper_state = 1.0 if gripper_cmd > 0.5 else 0.0
                    last_gripper_cmd = gripper_cmd
                    gripper_busy = True

                    def _do_gripper(cmd):
                        nonlocal gripper_busy, ctrl
                        ctrl.set_control(np.zeros(7))
                        arm.panda.stop_controller()
                        if cmd > 0.5:
                            arm.gripper_open()
                        else:
                            arm.gripper_close()
                        ctrl = controllers.IntegratedVelocity()
                        arm.panda.start_controller(ctrl)
                        ctrl.set_control(np.zeros(7))
                        gripper_busy = False

                    threading.Thread(
                        target=_do_gripper, args=(gripper_cmd,), daemon=True
                    ).start()

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
