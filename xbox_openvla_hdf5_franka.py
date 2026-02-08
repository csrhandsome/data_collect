#!/usr/bin/env python3
"""
Control Franka with a gamepad via pygame.
"""

import argparse
import os
import time
import threading
from datetime import datetime
from typing import Optional

import h5py
import numpy as np

from pygame_gamepad import PygameGamepadTeleop
from robotic_arm_controller import (
    RoboticArmControler,
    _LatestFrameBuffer,
    _camera_capture_worker,
)
from realsense_connector import RealSenseConnector


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Control Franka with a gamepad")
    parser.add_argument(
        "--control-frequency",
        type=float,
        default=10.0,
        help="Control loop frequency (Hz)",
    )
    parser.add_argument(
        "--translation-speed",
        type=float,
        default=0.005,
        help="Max translation step per loop (m)",
    )
    parser.add_argument(
        "--rotation-speed",
        type=float,
        default=0.1,
        help="Max rotation step per loop (rad)",
    )
    parser.add_argument(
        "--action-epsilon",
        type=float,
        default=1e-6,
        help=(
            "Clamp action deltas with abs<=epsilon to 0 for control/logging; "
            "set 0 to disable."
        ),
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable data logging (default: enabled)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Language instruction for data logging",
    )
    parser.add_argument(
        "--camera-timeout-ms",
        type=int,
        default=1000,
        help="Camera update timeout in ms",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=3600.0,
        help="Max logging duration (seconds)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Franka gamepad control")
    print("=" * 70)
    print(f"Control frequency: {args.control_frequency} Hz")
    print(f"Translation speed: {args.translation_speed} m/step")
    print(f"Rotation speed: {args.rotation_speed} rad/step")
    print(f"Action epsilon: {args.action_epsilon}")
    enable_logging = not args.no_logging
    if enable_logging:
        print(f"Data logging: enabled (max {args.max_duration} s)")
        print(f"Instruction: {args.instruction}")
    else:
        print("Data logging: disabled")
    print("=" * 70)
    print()

    if enable_logging and not args.instruction.strip():
        raise ValueError("--instruction is required when logging is enabled")

    print("Connecting gamepad (pygame) backend...")
    teleop = PygameGamepadTeleop()
    teleop.connect()
    print("Gamepad controller ready.\n")

    print("Initializing Franka arm...")
    arm = RoboticArmControler()

    camera = None
    if enable_logging:
        print("Initializing RealSense camera...")
        camera = RealSenseConnector()
        camera.connect()

    try:
        print("Opening gripper...")
        arm.gripper_open()

        print("Moving to start position...")
        arm.move_to_start()

        print("\nInitialization complete.")
        print("\nControl mapping:")
        print("  Left stick: XY translation")
        print("  L2/R2: Z down/up")
        print("  L1/R1: roll rotation")
        print("  Right stick: yaw/pitch rotation")
        print("  A: gripper close")
        print("  B: gripper open")
        print("  X/O: exit")
        print("  Ctrl+C: exit")
        print()

        _xbox_control(
            arm=arm,
            teleop=teleop,
            camera=camera,
            control_frequency=args.control_frequency,
            translation_speed=args.translation_speed,
            rotation_speed=args.rotation_speed,
            action_epsilon=args.action_epsilon,
            enable_logging=enable_logging,
            max_logging_duration=args.max_duration,
            camera_timeout_ms=args.camera_timeout_ms,
            language_instruction=args.instruction,
        )

    except KeyboardInterrupt:
        print("\n\nCtrl+C detected, exiting...")
    except Exception as exc:
        print(f"\n\nError: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        if hasattr(teleop, "disconnect"):
            teleop.disconnect()
        if camera is not None:
            camera.close()
        arm.cleanup()
        print("Cleanup complete.")


def _xbox_control(
    *,
    arm: RoboticArmControler,
    teleop: PygameGamepadTeleop,
    camera: Optional[RealSenseConnector],
    control_frequency: float,
    translation_speed: float,
    rotation_speed: float,
    action_epsilon: float,
    enable_logging: bool,
    max_logging_duration: float,
    camera_timeout_ms: int,
    language_instruction: str,
) -> None:
    """Gamepad control loop using pygame input."""
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
        r = R.from_euler('xyz', [roll, pitch, yaw])
        return r.as_quat()  # 返回 [x,y,z,w]

    def quat_inverse(q):
        """四元数求逆，输入输出都是 [x,y,z,w] 格式"""
        r = R.from_quat(q)
        return r.inv().as_quat()

    def quat_to_euler(q):
        """四元数转欧拉角，输入 [x,y,z,w] 格式"""
        r = R.from_quat(q)
        return r.as_euler('xyz')

    running = [True]

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
        log_file = os.path.join("./data/hdf5", f"gamepad_control_{timestamp}.h5")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        h5_file = h5py.File(log_file, "w")
        arm._last_log_filename = log_file

        h5_file.attrs["created_at"] = timestamp
        h5_file.attrs["control_mode"] = "gamepad"
        h5_file.attrs["control_frequency_hz"] = float(control_frequency)
        h5_file.attrs["translation_speed"] = float(translation_speed)
        h5_file.attrs["rotation_speed"] = float(rotation_speed)
        h5_file.attrs["action_epsilon"] = float(action_epsilon)
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
            raise ValueError("Gamepad logging requires camera=RealSenseConnector().")

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

        print("[Camera] Waiting for first frame...")
        max_wait_time = 5.0
        wait_start = time.time()
        while camera_buf.get_latest() is None:
            if time.time() - wait_start > max_wait_time:
                raise RuntimeError(
                    "Camera timeout: Failed to get first frame after 5 seconds."
                )
            time.sleep(0.01)
        print("[Camera] First frame acquired, ready to record!")

    img_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    instr_buf: list[str] = []
    flush_every = 20

    recording_started = False
    recording_stopped = False
    recording_started_at: Optional[float] = None

    target_position = arm.panda.get_position().astype(np.float64)
    target_orientation = arm.panda.get_orientation().astype(np.float64)
    gripper_state = 1.0

    workspace_half_extent = np.array([0.3, 0.3, 0.3], dtype=np.float64)
    workspace_min = target_position - workspace_half_extent - workspace_half_extent
    workspace_max = target_position + workspace_half_extent + workspace_half_extent
    max_ee_step_m = 0.05

    if h5_file is not None:
        h5_file.attrs["initial_position"] = target_position.tolist()
        h5_file.attrs["initial_orientation"] = target_orientation.tolist()
        print(f"[Data] Initial position: {target_position}")
        print(f"[Data] Initial orientation: {target_orientation}")

    arm.panda.start_controller(ctrl)
    time.sleep(2.0)

    last_gripper_cmd = 1.0
    has_joystick = getattr(teleop, "joystick", None) is not None
    motion_start_threshold = max(float(action_epsilon), 1e-3)

    reflex_error_occurred = False
    try:
        with arm.panda.create_context(frequency=control_frequency) as ctx:
            while ctx.ok() and running[0]:
                if (
                    enable_logging
                    and recording_started
                    and recording_started_at is not None
                    and (time.time() - recording_started_at) > max_logging_duration
                ):
                    print("\n[Recording] Max duration reached, stopping...")
                    running[0] = False
                    break

                action = teleop.get_action()
                if not action:
                    time.sleep(0.001)
                    continue

                axes, buttons = _get_gamepad_inputs(teleop)
                if buttons["x"] or buttons["o"]:
                    print("\n[Control] X/O pressed, exiting...")
                    running[0] = False
                    continue

                delta = np.zeros(6, dtype=np.float64)

                # Translation: left stick from teleop, triggers for Z
                delta[0] = float(action.get("delta_x", 0.0)) * translation_speed
                delta[1] = float(action.get("delta_y", 0.0)) * translation_speed
                delta[2] = (axes["rt"] - axes["lt"]) * translation_speed
                if not has_joystick:
                    delta[2] = float(action.get("delta_z", 0.0)) * translation_speed

                # Rotation: roll from shoulder buttons, yaw/pitch from right stick
                if buttons["l1"]:
                    delta[3] -= rotation_speed
                if buttons["r1"]:
                    delta[3] += rotation_speed
                delta[4] = (-axes["right_y"]) * rotation_speed
                delta[5] = (axes["right_x"]) * rotation_speed
                if not has_joystick:
                    delta[4] = 0.0
                    delta[5] = 0.0

                # Gripper control
                gripper_drift_delta = np.zeros(6, dtype=np.float64)
                gripper_cmd = last_gripper_cmd
                gripper_changed = False
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
                    target_orientation = arm.panda.get_orientation().astype(np.float64)  # [x,y,z,w]

                    gripper_drift_delta[:3] = target_position - pos_before
                    delta_quat_gripper = quat_multiply(target_orientation, quat_inverse(ori_before))  # [x,y,z,w]
                    gripper_drift_delta[3:] = quat_to_euler(delta_quat_gripper)  # 转为欧拉角

                    arm.panda.start_controller(ctrl)
                    last_gripper_cmd = gripper_cmd

                if np.any(gripper_drift_delta != 0):
                    delta = delta + gripper_drift_delta

                if action_epsilon > 0.0:
                    small_mask = np.abs(delta) <= action_epsilon
                    if np.any(small_mask):
                        delta = delta.copy()
                        delta[small_mask] = 0.0

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
                        print("[Warning] Camera frame is None, skipping", end="\r")
                        skip_frame = True
                    elif rgb224.shape != (224, 224, 3) or rgb224.dtype != np.uint8:
                        print(
                            f"[Warning] Invalid frame shape/dtype: {rgb224.shape}/{rgb224.dtype}",
                            end="\r",
                        )
                        skip_frame = True

                if skip_frame:
                    continue

                # Apply delta
                if np.any(delta != 0):
                    target_position_new = target_position + delta[:3]

                    step = target_position_new - target_position
                    step_norm = float(np.linalg.norm(step))
                    if step_norm > max_ee_step_m:
                        target_position_new = (
                            target_position + step / step_norm * max_ee_step_m
                        )

                    target_position_new = np.minimum(
                        np.maximum(target_position_new, workspace_min), workspace_max
                    )

                    target_orientation_new = target_orientation
                    if np.any(delta[3:] != 0):
                        # 欧拉角增量转为四元数，使用 [x,y,z,w] 格式
                        delta_quat = euler_to_quat(delta[3], delta[4], delta[5])  # [x,y,z,w]
                        target_orientation_new = quat_multiply(target_orientation, delta_quat)  # [x,y,z,w]

                    target_position = target_position_new
                    target_orientation = target_orientation_new

                    ctrl.set_control(target_position, target_orientation)  # 传入 [x,y,z,w] 格式

                if (
                    h5_file is not None
                    and camera_buf is not None
                    and image_ds is not None
                    and action_ds is not None
                    and instr_ds is not None
                    and not recording_stopped
                ):
                    motion_norm = float(np.linalg.norm(delta))
                    has_action = motion_norm >= motion_start_threshold

                    if (has_action or gripper_changed) and not recording_started:
                        recording_started = True
                        recording_started_at = time.time()
                        print("[Recording] First action detected, start logging...")

                    if recording_started:
                        action7 = np.concatenate(
                            [
                                delta.astype(np.float32, copy=False),
                                [np.float32(gripper_state)],
                            ]
                        )
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
                        print(f"[Recording] {n0 + batch_n} frames", end="\r")

    except RuntimeError as exc:
        error_msg = str(exc)
        if "cartesian_reflex" in error_msg or "motion aborted by reflex" in error_msg:
            print(f"\n[Error] Cartesian reflex triggered: {error_msg}")
            print("[Error] Data will NOT be saved due to reflex error")
            reflex_error_occurred = True
        raise
    except KeyboardInterrupt:
        print("\n[Recording] Ctrl+C detected, stopping...")
        running[0] = False
    finally:
        arm.panda.stop_controller()
        if camera_stop is not None:
            camera_stop.set()
        if camera_thread is not None:
            camera_thread.join(timeout=2.0)

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
                except Exception as exc:
                    print(f"[Cleanup] Failed to delete file: {exc}")
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
                print(f"\n[Recording] Finished. Total frames: {total_frames}")
                print(f"Saved to: {log_file}")

    print("\nGamepad control loop exited")


if __name__ == "__main__":
    main()
