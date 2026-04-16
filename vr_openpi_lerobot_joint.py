#!/usr/bin/env python3
"""
Collect Franka teleop data into a LeRobot dataset using VR (teleop_xr) input.

- Cameras: external + wrist (2x RealSense)
- Control: VR end-effector pose -> IK -> joint position controller
- Actions: absolute joint target + gripper state
- Output: LeRobot dataset (no intermediate HDF5)

Deadman: hold both VR triggers (long press) to enable arm movement.
Recording: start automatically when motion begins.
Y (left controller): save current episode and return to the start pose.
A (right controller): gripper close.  B: gripper open.

uv run vr_openpi_lerobot_joint.py \
  --instruction "Pick up the wide-mouth bottle" \
  --external-camera-serial 825412070292 \
  --wrist-camera-serial 825412070487 \
  --color-only

uv run vr_openpi_lerobot_joint.py \
  --instruction "Place the object into the basket" \
  --external-camera-serial 825412070292 \
  --wrist-camera-serial 825412070487 \
  --color-only \
  --date "4_1_final"
"""

import threading
import time
from pathlib import Path
from typing import Optional
import sys
import numpy as np

from control.collect_args import build_vr_lerobot_joint_parser
from control.vr_input import VRInputProcess
from control.vr_input_mapper import VREEPoseMapper
from control.dual_camera_manager import DualRealsenseManager
from control.util.lerobot_util import (
    _discard_unsaved_episode,
    _load_or_create_dataset,
    _prepare_episode_for_save,
)
from ik_solver import FrankaJointIKSolver
from control.robotic_arm_controller import RoboticArmControler


def main() -> None:
    parser = build_vr_lerobot_joint_parser()
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
    if enable_logging:
        args.repo_id = f"{args.repo_id}_{args.date}"
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
        _hold_current_joint_position(controller)
        if settle_s > 0.0:
            time.sleep(settle_s)
        return controller

    def _move_robot_to_start_pose() -> None:
        arm.move_to_start()
        if not arm.wait_until_stopped():
            max_vel = float(np.max(np.abs(np.asarray(arm.panda.get_state().dq))))
            print(
                "[Warning] Robot did not fully stop after move_to_start: "
                f"max_vel={max_vel:.4f} rad/s"
            )

    print("Opening gripper...")
    # arm.safe_open()
    arm.gripper_open()
    print("Moving to start position...")
    _move_robot_to_start_pose()
    ctrl = _start_joint_position_controller(settle_s=0.5)
    hold_qpos_target = _current_qpos().copy()

    active_instruction = args.instruction
    gripper_state = 1.0
    last_gripper_cmd = 1.0

    recording_started = False
    recording_started_at: Optional[float] = None
    frame_count = 0
    motion_start_threshold = max(float(args.action_epsilon), 1e-3)
    last_gripper_switch_time = 0.0
    gripper_busy = False
    gripper_switch_cooldown_s = 0.12
    reflex_error_occurred = False
    prev_y_pressed = False
    last_ik_warning_at = 0.0

    def _reset_robot_to_start() -> None:
        nonlocal ctrl, gripper_state, last_gripper_cmd, hold_qpos_target

        _hold_current_joint_position(ctrl)
        arm.panda.stop_controller()
        # arm.safe_open()
        arm.gripper_open()
        gripper_state = 1.0
        last_gripper_cmd = 1.0
        print("[Control] Moving to start position...")
        _move_robot_to_start_pose()
        ctrl = _start_joint_position_controller(settle_s=0.0)
        hold_qpos_target = _current_qpos().copy()
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
        _reset_robot_to_start()

    print("\nControl mapping (VR):")
    print("  Hold both triggers (long press): enable arm movement")
    print(
        "  Right controller pose: snapshot-based EE pose control -> IK -> joint position"
    )
    print("  Release / re-hold triggers: re-anchor the VR neutral pose")
    print("  A (right): gripper close | B (right): gripper open")
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
                    if gripper_busy:
                        continue
                    _finish_episode(
                        save_episode=frame_count > 0,
                        message="\n[Recording] Max duration reached, saving episode and returning to start...",
                    )
                    continue

                vr = vr_reader.latest

                y_pressed = bool(getattr(vr, "y_pressed", False))
                y_edge = y_pressed and not prev_y_pressed
                prev_y_pressed = y_pressed

                if y_edge:
                    if not enable_logging:
                        print(
                            "\n[Recording] Ignored Y press because logging is disabled."
                        )
                    elif gripper_busy:
                        print("\n[Recording] Ignored Y press because gripper is busy.")
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
                qpos_cmd = hold_qpos_target.copy()
                qvel_cmd = np.zeros(7, dtype=np.float64)
                has_joint_motion_cmd = False

                if vr.arm_enabled and not gripper_busy:
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

                has_joint_motion_cmd = bool(np.any(np.abs(joint_delta) > 0.0))

                if not gripper_busy:
                    if has_joint_motion_cmd:
                        qpos_cmd = np.clip(
                            qpos + joint_delta,
                            joint_limits[:, 0],
                            joint_limits[:, 1],
                        )
                        hold_qpos_target = qpos_cmd.copy()
                    else:
                        qpos_cmd = hold_qpos_target.copy()

                    if has_joint_motion_cmd and args.joint_velocity_limit > 0.0:
                        qvel_cmd = np.clip(
                            joint_delta * float(args.control_frequency),
                            -float(args.joint_velocity_limit),
                            float(args.joint_velocity_limit),
                        )
                    else:
                        qvel_cmd = np.zeros(7, dtype=np.float64)
                    ctrl.set_control(qpos_cmd, qvel_cmd)

                # --- Gripper ---
                gripper_cmd = last_gripper_cmd
                if vr.gripper_close:
                    gripper_cmd = 0.0
                elif vr.gripper_open:
                    gripper_cmd = 1.0

                now = time.time()
                if now - last_gripper_switch_time < gripper_switch_cooldown_s:
                    gripper_cmd = last_gripper_cmd

                gripper_changed = False
                if gripper_cmd != last_gripper_cmd and not gripper_busy:
                    gripper_changed = True
                    last_gripper_switch_time = now
                    gripper_state = 1.0 if gripper_cmd > 0.5 else 0.0
                    last_gripper_cmd = gripper_cmd
                    gripper_busy = True

                    def _do_gripper(cmd):
                        nonlocal gripper_busy, ctrl, hold_qpos_target
                        try:
                            _hold_current_joint_position(ctrl)
                            arm.panda.stop_controller()
                            if cmd > 0.5:
                                arm.gripper_open()
                            else:
                                arm.gripper_close()
                            ctrl = _start_joint_position_controller()
                            hold_qpos_target = _current_qpos().copy()
                            vr_mapper.reset()
                        finally:
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
