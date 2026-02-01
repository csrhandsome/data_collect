#!/usr/bin/env python3
"""
Collect Franka teleop data into a LeRobot dataset using DROID-style keys.

- Cameras: external + wrist (2x RealSense)
- Actions: measured joint velocity (rad/s) + gripper state
- Output: LeRobot dataset (no intermediate HDF5)

python xbox_openpi_lerobot_franka.py \
  --instruction "pick up the black cube" \
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
        help="Base repo id (timestamp is appended when logging).",
    )
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--control-frequency", type=float, default=15.0)
    parser.add_argument("--translation-speed", type=float, default=0.005)
    parser.add_argument("--rotation-speed", type=float, default=0.1)
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

    enable_logging = not args.no_logging
    if enable_logging and not args.instruction.strip():
        raise ValueError("--instruction is required when logging is enabled")

    print("=" * 70)
    print("Franka LeRobot data collection (joint velocity actions)")
    print("=" * 70)
    print(f"Control frequency: {args.control_frequency} Hz")
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.repo_id = f"{args.repo_id}_{timestamp}"
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
        dataset = _create_dataset(
            args.repo_id,
            fps=args.control_frequency,
            image_hw=args.image_hw,
            root=dataset_root,
        )
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
    from transforms3d.euler import euler2mat
    from transforms3d.quaternions import mat2quat
    from transforms3d.quaternions import qinverse
    from transforms3d.quaternions import qmult
    from transforms3d.euler import quat2euler

    ctrl = controllers.CartesianImpedance(filter_coeff=1.0)
    arm.panda.start_controller(ctrl)
    time.sleep(1.0)

    target_position = arm.panda.get_position().astype(np.float64)
    target_orientation = arm.panda.get_orientation().astype(np.float64)
    gripper_state = 1.0
    last_gripper_cmd = 1.0

    motion_start_threshold = max(float(args.action_epsilon), 1e-3)
    recording_started = False
    recording_started_at: Optional[float] = None
    frame_count = 0

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
                        _prepare_episode_for_save(dataset)
                        dataset.save_episode()
                        print(f"[Recording] Saved episode with {frame_count} frames")
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
                        )
                        arm.panda.start_controller(ctrl)
                        continue

                    print("\n[Control] X/O pressed, stopping...")
                    break

                delta = np.zeros(6, dtype=np.float64)
                delta[0] = float(action.get("delta_x", 0.0)) * args.translation_speed
                delta[1] = float(action.get("delta_y", 0.0)) * args.translation_speed
                delta[2] = (axes["rt"] - axes["lt"]) * args.translation_speed
                if getattr(teleop, "joystick", None) is None:
                    delta[2] = (
                        float(action.get("delta_z", 0.0)) * args.translation_speed
                    )

                if buttons["l1"]:
                    delta[3] -= args.rotation_speed
                if buttons["r1"]:
                    delta[3] += args.rotation_speed
                delta[4] = (-axes["right_y"]) * args.rotation_speed
                delta[5] = (axes["right_x"]) * args.rotation_speed

                gripper_changed = False
                gripper_cmd = last_gripper_cmd
                if buttons["a"]:
                    gripper_cmd = 0.0
                elif buttons["b"]:
                    gripper_cmd = 1.0

                if gripper_cmd != last_gripper_cmd:
                    gripper_changed = True
                    pos_before = arm.panda.get_position().astype(np.float64)
                    ori_before = arm.panda.get_orientation().astype(np.float64)

                    arm.panda.stop_controller()
                    if gripper_cmd > 0.5:
                        arm.gripper_open()
                        gripper_state = 1.0
                    else:
                        arm.gripper_close()
                        gripper_state = 0.0

                    target_position = arm.panda.get_position().astype(np.float64)
                    target_orientation = arm.panda.get_orientation().astype(np.float64)

                    drift = np.zeros(6, dtype=np.float64)
                    drift[:3] = target_position - pos_before
                    delta_quat = qmult(target_orientation, qinverse(ori_before))
                    drift[3:] = quat2euler(delta_quat, axes="sxyz")

                    arm.panda.start_controller(ctrl)
                    last_gripper_cmd = gripper_cmd

                    if np.any(drift != 0):
                        delta = delta + drift

                if args.action_epsilon > 0.0:
                    small_mask = np.abs(delta) <= args.action_epsilon
                    if np.any(small_mask):
                        delta = delta.copy()
                        delta[small_mask] = 0.0

                if np.any(delta != 0):
                    target_position_new = target_position + delta[:3]
                    target_orientation_new = target_orientation
                    if np.any(delta[3:] != 0):
                        delta_rot = euler2mat(delta[3], delta[4], delta[5])
                        delta_quat = mat2quat(delta_rot)
                        target_orientation_new = qmult(target_orientation, delta_quat)

                    target_position = target_position_new
                    target_orientation = target_orientation_new
                    ctrl.set_control(target_position, target_orientation)

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

        if enable_logging and dataset is not None:
            try:
                if frame_count > 0:
                    _prepare_episode_for_save(dataset)
                    dataset.save_episode()
                    print(f"\n[Recording] Saved episode with {frame_count} frames")
            finally:
                try:
                    dataset.stop_image_writer()
                except Exception:
                    pass
                try:
                    dataset.finalize()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
