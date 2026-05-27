#!/usr/bin/env python3
"""
Collect Franka teleop data into a LeRobot dataset using DROID-style keys.

- Cameras: external + wrist (2x RealSense)
- Control: joint velocity via IntegratedVelocity + DROID dm_control IK solver
- Actions: normalized joint velocity [-1, 1] + gripper state
- Output: LeRobot dataset (no intermediate HDF5)

uv run xbox_openpi_lerobot_franka.py \
  --instruction "Pick up the wide-mouth bottle" \
  --second-instruction "..." \
  --external-camera-serial 825412070292 \
  --wrist-camera-serial 825412070487 \
  --color-only

Controls:
- X: switch to the second prompt.
- Y: start/stop recording (save on stop, no reset).
- A/B: gripper close/open.
"""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from control.collect_args import build_xbox_lerobot_droid_parser
from control.pygame_gamepad import PygameGamepadTeleop
from control.dual_camera_manager import DualRealsenseManager
from control.util.lerobot_util import (
    _discard_unsaved_episode,
    _load_or_create_dataset,
    _prepare_episode_for_save,
)
from control.ik_solver.dm_control_ik_solver import DroidIKSolver
from control.robotic_arm_controller import RoboticArmControler


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
        "y": False,
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
    buttons["y"] = read_button(3)
    buttons["o"] = read_button(3)

    return axes, buttons


def main() -> None:
    parser = build_xbox_lerobot_droid_parser()
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
        date = "2_24"
        args.repo_id = f"{args.repo_id}_{date}"
        print(f"Logging: enabled (max {args.max_duration} s)")
        print(f"Instruction: {args.instruction}")
        if args.second_instruction.strip():
            print(f"Second instruction: {args.second_instruction}")
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

    print("Opening gripper...")
    arm.gripper_open()
    print("Moving to start position...")
    arm.move_to_start()
    if not arm.wait_until_stopped():
        max_vel = float(np.max(np.abs(np.asarray(arm.panda.get_state().dq))))
        print(
            "[Warning] Robot did not fully stop after move_to_start: "
            f"max_vel={max_vel:.4f} rad/s"
        )

    # --- DROID IK solver (replaces simple Jacobian pseudoinverse) ---
    print("Initializing DROID IK solver (dm_control)...")
    ik_solver = DroidIKSolver(control_hz=float(args.control_frequency))

    from panda_py import controllers

    ctrl = controllers.IntegratedVelocity()
    arm.panda.start_controller(ctrl)
    time.sleep(0.5)
    ctrl.set_control(np.zeros(7))

    active_instruction = args.instruction
    gripper_state = 1.0
    last_gripper_cmd = 1.0

    recording_started = False
    recording_started_at: Optional[float] = None
    frame_count = 0
    last_gripper_switch_time = 0.0
    gripper_busy = False
    gripper_switch_cooldown_s = 0.12
    reflex_error_occurred = False
    prev_x_pressed = False
    prev_y_pressed = False

    def _finish_recording(*, save_episode: bool, message: str) -> None:
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
                        _discard_unsaved_episode(dataset)
                    except Exception:
                        pass
            else:
                try:
                    _discard_unsaved_episode(dataset)
                except Exception:
                    pass

        frame_count = 0
        recording_started = False
        recording_started_at = None

    print("\nControl mapping:")
    print("  Left stick: XY translation")
    print("  L2/R2: Z down/up")
    print("  L1/R1: roll rotation")
    print("  Right stick: yaw/pitch rotation")
    print("  A: gripper close | B: gripper open")
    print("  X: switch to second prompt")
    print(
        "  Y (top face button): start/stop recording (save on stop, no move_to_start)"
    )

    try:
        with arm.panda.create_context(frequency=args.control_frequency) as ctx:
            while ctx.ok():
                if (
                    enable_logging
                    and recording_started
                    and recording_started_at is not None
                    and (time.time() - recording_started_at) > args.max_duration
                ):
                    ctrl.set_control(np.zeros(7))
                    _finish_recording(
                        save_episode=frame_count > 0,
                        message="\n[Recording] Max duration reached, stopping current episode...",
                    )
                    continue

                action = teleop.get_action()
                if not action:
                    time.sleep(0.001)
                    continue

                axes, buttons = _get_gamepad_inputs(teleop)
                x_pressed = buttons["x"]
                y_pressed = buttons["y"]
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
                    elif not recording_started:
                        recording_started = True
                        recording_started_at = time.time()
                        print(
                            f"\n[Recording] Y pressed, start logging with prompt: {active_instruction}"
                        )
                    else:
                        ctrl.set_control(np.zeros(7))
                        _finish_recording(
                            save_episode=frame_count > 0,
                            message=(
                                "\n[Control] Y pressed, stopping and saving episode..."
                                if frame_count > 0
                                else "\n[Control] Y pressed, stopping recording with no captured frames."
                            ),
                        )
                    continue

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
                if not gripper_busy:
                    ctrl.set_control(joint_delta)

                # --- Gripper ---
                gripper_cmd = last_gripper_cmd
                if buttons["a"]:
                    gripper_cmd = 0.0
                elif buttons["b"]:
                    gripper_cmd = 1.0

                now = time.time()
                if now - last_gripper_switch_time < gripper_switch_cooldown_s:
                    gripper_cmd = last_gripper_cmd

                if gripper_cmd != last_gripper_cmd and not gripper_busy:
                    last_gripper_switch_time = now
                    gripper_state = 1.0 if gripper_cmd > 0.5 else 0.0
                    last_gripper_cmd = gripper_cmd
                    gripper_busy = True

                    def _do_gripper(cmd):
                        nonlocal gripper_busy, ctrl
                        # 同步操作放到线程里面
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

                external_img, wrist_img = camera_manager.get_images()
                if external_img is None or wrist_img is None:
                    continue

                if external_img.shape != (args.image_hw, args.image_hw, 3):
                    continue
                if wrist_img.shape != (args.image_hw, args.image_hw, 3):
                    continue

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

        try:
            camera_manager.close()
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
                            _discard_unsaved_episode(dataset)
                        except Exception:
                            pass
            finally:
                try:
                    dataset.stop_image_writer()
                except Exception:
                    pass
        elif enable_logging and dataset is not None and reflex_error_occurred:
            try:
                _discard_unsaved_episode(dataset)
            except Exception:
                pass
            try:
                dataset.stop_image_writer()
            except Exception:
                pass


if __name__ == "__main__":
    main()
