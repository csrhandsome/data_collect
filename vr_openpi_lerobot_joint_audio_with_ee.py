#!/usr/bin/env python3
"""
Collect Franka teleop data into a LeRobot dataset using VR input, with episode audio.

- Cameras: external + wrist (2x RealSense)
- Audio: per-episode microphone WAV + audio metadata sidecar
- Control: VR end-effector pose -> Cartesian impedance controller
- Actions: next measured joint position + gripper state
- Output: LeRobot dataset + audio sync JSON

Deadman: hold both VR triggers (long press) to enable arm movement.
Recording: start automatically when motion begins.
Y (left controller): save current episode and return to the start pose.
A (right controller): gripper close.  B: gripper open.

uv run vr_openpi_lerobot_joint_audio_with_ee.py \
  --instruction "Place the object into the basket" \
  --external-camera-serial 825412070292 \
  --wrist-camera-serial 825412070487 \
  --color-only \
  --date "7_6_audio" \
  --label stop
如果ctrl c无法终止
ps -ef | grep vr_openpi_lerobot_joint_audio_with_ee
然后杀主进程和 uv 包装进程：
kill -TERM <pid1> <pid2>
还不退再用：
kill -KILL <pid1> <pid2>
"""

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import websockets
import websockets.sync.client

from control.collect_args import build_vr_lerobot_joint_audio_parser
from control.dual_camera_manager import DualRealsenseManager
from control.util.lerobot_util import (
    _discard_unsaved_episode,
    _load_or_create_dataset,
    _prepare_episode_for_save,
)
from control.microphone_connector import MicrophoneRecorder
from control.robotic_arm_controller import RoboticArmControler
from control.vr_input import VRInputProcess
from control.vr_input_mapper import VREEPoseMapper
from data_analysis.instruction_audio_window import make_pending_instruction_audio_window
from data_analysis.preprocess_vad import compute_vad_for_dataset


CUSTOM_START_JOINT_POSITION: tuple[float, ...] | None = (
    0.00388,
    -0.45916,
    -0.15237,
    -2.47266,
    -0.06778,
    1.96442,
    0.70214,
)


def _path_for_json(path: Optional[Path], root: Optional[Path] = None) -> Optional[str]:
    if path is None:
        return None
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


class ReactiveDeskVlaClient:
    """Minimal Reactive Desk websocket client compatible with the openpi sender."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        path: str = "/ws/VlaIngest",
        *,
        enabled: bool = True,
    ) -> None:
        self._uri = self._build_ws_uri(host, port, path)
        self._enabled = enabled
        self._ws: websockets.sync.client.ClientConnection | None = None

    def connect(self) -> None:
        if not self._enabled or self._ws is not None:
            return

        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
        )
        self._ws.recv()

    def send_predictions(
        self,
        xyz: np.ndarray,
        probabilities: np.ndarray,
        *,
        prompt: str = "",
        is_executing: bool = True,
    ) -> bool:
        if not self._enabled:
            return False

        xyz = np.asarray(xyz, dtype=np.float32)
        probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)

        predictions = [
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]) if point.shape[0] > 2 else 0.0,
                "probability": float(probabilities[rank])
                if rank < probabilities.shape[0]
                else 1.0,
                "rank": int(rank),
            }
            for rank, point in enumerate(xyz)
        ]
        payload = {
            "type": "vla_predictions",
            "predictions": predictions,
            "is_executing": is_executing,
            "current_prompt": prompt,
        }

        try:
            self.connect()
            if self._ws is None:
                return False
            self._ws.send(json.dumps(payload))
            self._ws.recv()
            return True
        except websockets.ConnectionClosed:
            self._ws = None
            return False
        except OSError:
            self._ws = None
            return False

    def close(self) -> None:
        if self._ws is None:
            return
        self._ws.close()
        self._ws = None

    @staticmethod
    def _build_ws_uri(host: str, port: int, path: str) -> str:
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        if path:
            path = path if path.startswith("/") else f"/{path}"
            if not uri.endswith(path):
                uri = f"{uri.rstrip('/')}{path}"
        return uri


def main() -> None:
    parser = build_vr_lerobot_joint_audio_parser()
    for action in parser._actions:
        if action.dest in {
            "ik_solver",
            "ik_attempts",
            "max_joint_delta",
            "joint_velocity_limit",
        }:
            action.help = argparse.SUPPRESS
    parser.add_argument(
        "--ee-filter-coeff",
        type=float,
        default=0.35,
        help="CartesianImpedance input filter coefficient. 1 disables filtering; smaller is smoother but slower.",
    )
    parser.add_argument(
        "--ee-nullspace-stiffness",
        type=float,
        default=0.5,
        help="CartesianImpedance nullspace stiffness.",
    )
    parser.add_argument("--reactive-desk-host", type=str, default="127.0.0.1")
    parser.add_argument("--reactive-desk-port", type=int, default=8000)
    parser.add_argument("--reactive-desk-path", type=str, default="/ws/VlaIngest")
    parser.add_argument(
        "--no-reactive-desk",
        dest="reactive_desk_enabled",
        action="store_false",
        default=True,
        help="Disable sending current end-effector xy to Reactive Desk.",
    )
    parser.add_argument(
        "--label",
        choices=("straight", "detour", "none", "stop"),
        default="none",
        help="Route label for this episode.",
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
    if not 0.0 < args.ee_filter_coeff <= 1.0:
        raise ValueError("--ee-filter-coeff must be in (0, 1]")
    if args.ee_nullspace_stiffness < 0:
        raise ValueError("--ee-nullspace-stiffness must be >= 0")
    if args.audio_sample_rate <= 0:
        raise ValueError("--audio-sample-rate must be > 0")
    if args.audio_channels <= 0:
        raise ValueError("--audio-channels must be > 0")
    if args.reactive_desk_port <= 0:
        raise ValueError("--reactive-desk-port must be > 0")

    enable_logging = not args.no_logging
    if enable_logging and not args.instruction.strip():
        raise ValueError("--instruction is required when logging is enabled")

    label = str(args.label)

    print("=" * 70)
    print("Franka LeRobot data collection (VR teleop + audio)")
    print("=" * 70)
    trans_limit_str = (
        "off" if args.max_ee_translation <= 0 else f"±{args.max_ee_translation:.3f}m"
    )
    rot_limit_str = (
        "off" if args.max_ee_rotation <= 0 else f"±{args.max_ee_rotation:.3f}rad"
    )
    print(f"Control frequency: {args.control_frequency} Hz")
    print(f"Sensitivity: {args.sensitivity}")
    print(
        "EE control tuning: "
        f"step_xyz={args.max_ee_translation_step:.3f}m, "
        f"step_rot={args.max_ee_rotation_step:.3f}rad, "
        f"limit_xyz={trans_limit_str}, "
        f"limit_rot={rot_limit_str}, "
        f"vr_rot={'on' if args.vr_enable_rotation else 'off'}, "
        f"filter={args.ee_filter_coeff:.2f}, "
        f"nullspace={args.ee_nullspace_stiffness:.2f}"
    )
    print("Action logging: next measured 7D joint position + gripper")
    print(
        "VR input smoothing: "
        f"pos_alpha={args.vr_position_alpha:.2f}, "
        f"rot_alpha={args.vr_rotation_alpha:.2f}"
    )
    print(
        f"Camera stream: {args.camera_width}x{args.camera_height}@{args.camera_fps} "
        f"(depth={'off' if args.color_only else 'on'})"
    )
    print(
        f"Audio stream: {args.audio_sample_rate} Hz, "
        f"{args.audio_channels} channel(s), int16 PCM"
    )
    print(f"VR server: {args.vr_host}:{args.vr_port}")
    print(
        "Reactive Desk: "
        + (
            f"{args.reactive_desk_host}:{args.reactive_desk_port}{args.reactive_desk_path}"
            if args.reactive_desk_enabled
            else "disabled"
        )
    )
    print(f"External camera serial: {args.external_camera_serial}")
    print(f"Wrist camera serial: {args.wrist_camera_serial}")
    if enable_logging:
        args.repo_id = f"{args.repo_id}_{args.date}"
        print(f"Logging: enabled (max {args.max_duration} s)")
        print(f"Instruction: {args.instruction}")
        print(f"Label: {label}")
        print(f"LeRobot repo_id: {args.repo_id}")
    else:
        print("Logging: disabled")
    print("=" * 70)

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

    vr_mapper = VREEPoseMapper(
        translation_scale=args.vr_translation_scale,
        rotation_scale=args.vr_rotation_scale,
        max_translation_step=max_ee_translation_step,
        max_rotation_step=max_ee_rotation_step,
        translation_limit=max_ee_translation,
        rotation_limit=max_ee_rotation,
        sensitivity=args.sensitivity,
        enable_rotation=args.vr_enable_rotation,
    )

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
    dataset_root: Optional[Path] = None
    audio_dir: Optional[Path] = None
    microphone_recorder: Optional[MicrophoneRecorder] = None
    if enable_logging:
        dataset_root = Path(__file__).resolve().parent / "data" / args.repo_id
        resume_existing = dataset_root.exists()
        dataset = _load_or_create_dataset(
            args.repo_id,
            fps=args.control_frequency,
            image_hw=args.image_hw,
            root=dataset_root,
        )
        audio_dir = dataset_root / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        microphone_recorder = MicrophoneRecorder(
            sample_rate=int(args.audio_sample_rate),
            channels=int(args.audio_channels),
            dtype=np.int16,
        )
        microphone_recorder.start()
        if resume_existing:
            print(f"LeRobot dataset exists, resuming: {dataset_root}")
        else:
            print(f"LeRobot dataset path: {dataset_root}")
        print(f"Audio path: {audio_dir}")

    print("[Camera] Waiting for first frames...")
    camera_manager.wait_for_frames(timeout_s=float(args.camera_startup_timeout_s))
    print("[Camera] First frames acquired, ready to record!")

    from panda_py import controllers

    def _current_qpos() -> np.ndarray:
        robot_state = arm.panda.get_state()
        return np.asarray(robot_state.q, dtype=np.float64)

    def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
        return np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )

    def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
        return np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float64,
        )

    def _quat_angle_xyzw(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a / max(np.linalg.norm(a), 1e-12)
        b = b / max(np.linalg.norm(b), 1e-12)
        dot = abs(float(np.dot(a, b)))
        return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

    def _current_ee_pose() -> tuple[np.ndarray, np.ndarray]:
        return (
            arm.panda.get_position().astype(np.float64),
            arm.panda.get_orientation().astype(np.float64),
        )

    def _hold_current_ee_pose(controller) -> None:
        ee_pos, ee_quat_xyzw = _current_ee_pose()
        qpos = _current_qpos()
        controller.set_control(ee_pos, ee_quat_xyzw, qpos)

    def _start_ee_controller(settle_s: float = 0.0):
        controller = controllers.CartesianImpedance(
            filter_coeff=float(args.ee_filter_coeff),
            nullspace_stiffness=float(args.ee_nullspace_stiffness),
        )
        arm.panda.start_controller(controller)
        _hold_current_ee_pose(controller)
        if settle_s > 0.0:
            time.sleep(settle_s)
        return controller

    def _format_joint_position(qpos: np.ndarray) -> str:
        return np.array2string(
            np.asarray(qpos, dtype=np.float64),
            precision=5,
            separator=", ",
            suppress_small=False,
        )

    def _print_current_joint_position(qpos: np.ndarray) -> None:
        line = f"[Joint] current q = {_format_joint_position(qpos)}"
        print(f"{line:<140}", end="\r", flush=True)

    def _move_robot_to_joint_pose(
        target_qpos: np.ndarray | list[float] | tuple[float, ...],
    ) -> None:
        qpos = np.asarray(target_qpos, dtype=np.float64).flatten()
        if qpos.shape != (7,):
            raise ValueError(f"target_qpos must be 7D, got shape {qpos.shape}")
        if arm.auto_set_default_behavior:
            arm.panda.set_default_behavior()
        print(f"[Control] Moving to custom joint pose: {_format_joint_position(qpos)}")
        arm.panda.move_to_joint_position(qpos, speed_factor=arm.joint_speed_factor)

    def _move_robot_to_start_pose() -> None:
        move_name = "move_to_start"
        if CUSTOM_START_JOINT_POSITION is None:
            arm.move_to_start()
        else:
            move_name = "custom start joint pose"
            _move_robot_to_joint_pose(CUSTOM_START_JOINT_POSITION)
        if not arm.wait_until_stopped():
            max_vel = float(np.max(np.abs(np.asarray(arm.panda.get_state().dq))))
            print(
                f"[Warning] Robot did not fully stop after {move_name}: "
                f"max_vel={max_vel:.4f} rad/s"
            )

    print("Opening gripper...")
    arm.gripper_open()
    print("Moving to start position...")
    _move_robot_to_start_pose()
    ctrl = _start_ee_controller(settle_s=0.5)
    hold_ee_pos_target, hold_ee_quat_xyzw_target = _current_ee_pose()
    hold_qpos_target = _current_qpos().copy()

    def _refresh_hold_target_from_current() -> None:
        nonlocal hold_ee_pos_target, hold_ee_quat_xyzw_target, hold_qpos_target
        hold_ee_pos_target, hold_ee_quat_xyzw_target = _current_ee_pose()
        hold_qpos_target = _current_qpos().copy()

    def _set_hold_control(controller) -> None:
        controller.set_control(
            hold_ee_pos_target,
            hold_ee_quat_xyzw_target,
            hold_qpos_target,
        )

    active_instruction = args.instruction
    reactive_desk_client = ReactiveDeskVlaClient(
        host=args.reactive_desk_host,
        port=args.reactive_desk_port,
        path=args.reactive_desk_path,
        enabled=args.reactive_desk_enabled,
    )
    gripper_state = 1.0
    last_gripper_cmd = 1.0

    recording_started = False
    recording_started_at: Optional[float] = None
    episode_start_monotonic_ns: Optional[int] = None
    current_episode_index: Optional[int] = None
    current_audio_path: Optional[Path] = None
    frame_records: list[dict] = []
    pending_frame: Optional[dict] = None
    frame_count = 0
    motion_start_threshold = max(float(args.action_epsilon), 1e-3)
    last_gripper_switch_time = 0.0
    gripper_busy = False
    gripper_switch_cooldown_s = 0.12
    reflex_error_occurred = False
    prev_y_pressed = False
    prev_arm_enabled = False

    def _reset_episode_state() -> None:
        nonlocal frame_count
        nonlocal recording_started
        nonlocal recording_started_at
        nonlocal episode_start_monotonic_ns
        nonlocal current_episode_index
        nonlocal current_audio_path
        nonlocal frame_records
        nonlocal pending_frame

        frame_count = 0
        recording_started = False
        recording_started_at = None
        episode_start_monotonic_ns = None
        current_episode_index = None
        current_audio_path = None
        frame_records = []
        pending_frame = None
        if microphone_recorder is not None:
            microphone_recorder.default_output_path = None
            if not microphone_recorder.is_recording:
                microphone_recorder.reset()

    def _append_pending_frame(action_qpos: np.ndarray) -> None:
        nonlocal frame_count, pending_frame

        if pending_frame is None or dataset is None:
            return

        action_gripper_state = float(pending_frame["action_gripper_state"])
        actions = np.concatenate(
            [
                np.asarray(action_qpos, dtype=np.float32),
                [np.float32(action_gripper_state)],
            ],
            dtype=np.float32,
        )

        frame_record = pending_frame["frame_record"]
        frame_record["action_joint_position"] = actions[:7].tolist()
        frame_record["action_gripper_position"] = action_gripper_state

        dataset.add_frame(
            {
                "exterior_image_1_left": pending_frame["external_img"],
                "exterior_image_2_left": pending_frame["blank"],
                "wrist_image_left": pending_frame["wrist_img"],
                "joint_position": pending_frame["joint_pos"],
                "gripper_position": pending_frame["gripper_pos"],
                "actions": actions,
                "task": active_instruction,
            }
        )
        frame_records.append(frame_record)
        frame_count += 1
        pending_frame = None
        if frame_count % 50 == 0:
            print(f"[Recording] {frame_count} frames", end="\r")

    def _cleanup_episode_files(*paths: Optional[Path]) -> None:
        for path in paths:
            if path is None:
                continue
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    def _write_episode_sync_json(audio_path: Path) -> Path:
        if (
            audio_dir is None
            or dataset_root is None
            or current_episode_index is None
            or microphone_recorder is None
        ):
            raise RuntimeError(
                "Episode sync JSON cannot be written without logging state"
            )

        sync_path = audio_dir / f"episode_{current_episode_index:06d}.sync.json"
        payload = {
            "episode_index": current_episode_index,
            "task": active_instruction,
            "label": label,
            "divergence_time": None,
            "instruction_audio_window": make_pending_instruction_audio_window(),
            "control_frequency": float(args.control_frequency),
            "audio_path": _path_for_json(audio_path, dataset_root),
            "audio_metadata_path": _path_for_json(
                microphone_recorder.last_metadata_path, dataset_root
            ),
            "audio_start_monotonic_ns": microphone_recorder.audio_start_monotonic_ns,
            "audio_stop_monotonic_ns": microphone_recorder.audio_stop_monotonic_ns,
            "episode_start_monotonic_ns": episode_start_monotonic_ns,
            "video_frames": frame_count,
            "frame_records": frame_records,
            "vad_metadata": {
                "processed": False,
                "stage": "offline_preprocess",
                "tool": "silero-vad",
                "note": (
                    "Run data_analysis/preprocess_vad.py after collection to fill "
                    "vad_segments and refresh instruction_audio_window."
                ),
            },
            "vad_segments": [],
            "camera_serials": {
                "external": camera_manager.external_camera.serial,
                "wrist": camera_manager.wrist_camera.serial,
            },
        }
        sync_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return sync_path

    def _finalize_episode_data(*, save_episode: bool) -> None:
        nonlocal frame_count

        if save_episode and pending_frame is not None:
            _append_pending_frame(_current_qpos())

        should_persist = (
            bool(save_episode)
            and frame_count > 0
            and dataset is not None
            and microphone_recorder is not None
            and current_episode_index is not None
            and current_audio_path is not None
        )

        audio_path: Optional[Path] = None
        audio_metadata_path: Optional[Path] = None
        sync_path: Optional[Path] = None

        if microphone_recorder is not None:
            microphone_recorder.default_output_path = (
                current_audio_path if should_persist else None
            )
            try:
                audio_path = microphone_recorder.stop_recording()
                audio_metadata_path = microphone_recorder.last_metadata_path
            except Exception as exc:
                print(f"[Error] Failed to stop microphone recorder: {exc}")
                should_persist = False

        if not should_persist:
            if dataset is not None:
                try:
                    _discard_unsaved_episode(dataset)
                except Exception:
                    pass
            _reset_episode_state()
            return

        try:
            if audio_path is None:
                audio_path = microphone_recorder.save_recording(current_audio_path)
                audio_metadata_path = microphone_recorder.last_metadata_path

            sync_path = _write_episode_sync_json(audio_path)
            _prepare_episode_for_save(dataset)
            dataset.save_episode()
            print(
                f"[Recording] Saved episode {current_episode_index:06d} with {frame_count} frames"
            )
            print(f"[Recording] Saved audio: {audio_path}")
            print(f"[Recording] Saved sync metadata: {sync_path}")
        except Exception as exc:
            print(f"[Error] Failed to save episode: {exc}")
            _cleanup_episode_files(audio_path, audio_metadata_path, sync_path)
            try:
                _discard_unsaved_episode(dataset)
            except Exception:
                pass
        finally:
            _reset_episode_state()

    def _reset_robot_to_start() -> None:
        nonlocal ctrl, gripper_state, last_gripper_cmd

        _set_hold_control(ctrl)
        arm.panda.stop_controller()
        arm.gripper_open()
        gripper_state = 1.0
        last_gripper_cmd = 1.0
        print("[Control] Moving to start position...")
        _move_robot_to_start_pose()
        ctrl = _start_ee_controller(settle_s=0.0)
        _refresh_hold_target_from_current()
        _set_hold_control(ctrl)
        vr_mapper.reset()

    def _finish_episode(*, save_episode: bool, message: str) -> None:
        print(message)
        _finalize_episode_data(save_episode=save_episode)
        _reset_robot_to_start()

    def _send_current_ee_xy(ee_pos: np.ndarray) -> None:
        reactive_desk_client.send_predictions(
            np.asarray(ee_pos[:2], dtype=np.float32).reshape(1, 2),
            np.ones(1, dtype=np.float32),
            prompt=active_instruction,
            is_executing=True,
        )

    print("\nControl mapping (VR):")
    print("  Hold both triggers (long press): enable arm movement")
    print("  Right controller pose: EE pose target -> Cartesian impedance control")
    print("  Action: next measured joint position + gripper")
    print("  Release / re-hold triggers: re-anchor the VR neutral pose")
    print("  A (right): gripper close | B (right): gripper open")
    print("  Recording: starts automatically when motion begins")
    print("  Audio: starts with the first recorded motion and saves per episode")
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
                        save_episode=frame_count > 0 or pending_frame is not None,
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
                            save_episode=frame_count > 0 or pending_frame is not None,
                            message=(
                                "\n[Control] Y pressed, saving episode and returning to start..."
                                if frame_count > 0 or pending_frame is not None
                                else "\n[Control] Y pressed, resetting with no captured frames."
                            ),
                        )
                    continue

                robot_state = arm.panda.get_state()
                qpos = np.asarray(robot_state.q, dtype=np.float64)
                _print_current_joint_position(qpos)
                if enable_logging and recording_started and pending_frame is not None:
                    _append_pending_frame(qpos)

                ee_pos, ee_quat_xyzw = _current_ee_pose()
                _send_current_ee_xy(ee_pos)

                arm_enabled = bool(vr.arm_enabled)
                if not arm_enabled and prev_arm_enabled:
                    _refresh_hold_target_from_current()
                    vr_mapper.reset()
                elif not arm_enabled:
                    vr_mapper.reset()
                prev_arm_enabled = arm_enabled

                if arm_enabled:
                    hold_ee_quat_wxyz = _quat_xyzw_to_wxyz(hold_ee_quat_xyzw_target)
                    target_ee_pos, target_ee_quat_wxyz = vr_mapper.map(
                        vr,
                        hold_ee_pos_target,
                        hold_ee_quat_wxyz,
                    )
                    target_ee_quat_xyzw = _quat_wxyz_to_xyzw(target_ee_quat_wxyz)
                else:
                    target_ee_pos = hold_ee_pos_target.copy()
                    target_ee_quat_xyzw = hold_ee_quat_xyzw_target.copy()

                command_translation = target_ee_pos - hold_ee_pos_target
                rotation_error = _quat_angle_xyzw(
                    target_ee_quat_xyzw, hold_ee_quat_xyzw_target
                )
                motion_norm = float(np.linalg.norm(command_translation))
                rotation_motion_threshold = max(float(args.action_epsilon), 1e-3)
                has_ee_motion_cmd = bool(
                    arm_enabled
                    and not gripper_busy
                    and (
                        motion_norm >= motion_start_threshold
                        or rotation_error >= rotation_motion_threshold
                    )
                )

                if not gripper_busy:
                    if has_ee_motion_cmd:
                        ctrl.set_control(target_ee_pos, target_ee_quat_xyzw, qpos)
                        hold_ee_pos_target = target_ee_pos.copy()
                        hold_ee_quat_xyzw_target = target_ee_quat_xyzw.copy()
                        hold_qpos_target = qpos.copy()
                    else:
                        _set_hold_control(ctrl)

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
                        nonlocal gripper_busy, ctrl
                        try:
                            _set_hold_control(ctrl)
                            arm.panda.stop_controller()
                            if cmd > 0.5:
                                arm.gripper_open()
                            else:
                                arm.gripper_close()
                            ctrl = _start_ee_controller()
                            _refresh_hold_target_from_current()
                            _set_hold_control(ctrl)
                            vr_mapper.reset()
                        finally:
                            gripper_busy = False

                    threading.Thread(
                        target=_do_gripper, args=(gripper_cmd,), daemon=True
                    ).start()

                if not enable_logging:
                    continue

                external_img, wrist_img, external_ts, wrist_ts = (
                    camera_manager.get_frames()
                )
                if (
                    external_img is None
                    or wrist_img is None
                    or external_ts is None
                    or wrist_ts is None
                ):
                    continue

                if external_img.shape != (args.image_hw, args.image_hw, 3):
                    continue
                if wrist_img.shape != (args.image_hw, args.image_hw, 3):
                    continue

                has_action = has_ee_motion_cmd or gripper_changed

                if has_action and not recording_started:
                    if (
                        dataset is None
                        or audio_dir is None
                        or microphone_recorder is None
                    ):
                        raise RuntimeError(
                            "Logging state is not initialized for audio capture"
                        )

                    current_episode_index = int(dataset.episode_buffer["episode_index"])
                    current_audio_path = (
                        audio_dir / f"episode_{current_episode_index:06d}.wav"
                    )
                    episode_start_monotonic_ns = time.monotonic_ns()
                    microphone_recorder.default_output_path = current_audio_path
                    if not microphone_recorder.start_recording():
                        microphone_recorder.default_output_path = None
                        current_episode_index = None
                        current_audio_path = None
                        episode_start_monotonic_ns = None
                        print(
                            "\n[Error] Failed to start microphone recording; skipping episode start."
                        )
                        continue

                    recording_started = True
                    recording_started_at = time.time()
                    frame_records = []
                    print(
                        "\n[Recording] First motion detected, start audio/video logging "
                        f"for episode {current_episode_index:06d} with prompt: {active_instruction}"
                    )

                if recording_started:
                    joint_pos = np.asarray(robot_state.q, dtype=np.float32)
                    gripper_pos = np.asarray(
                        [np.float32(gripper_state)], dtype=np.float32
                    )
                    blank = np.zeros_like(external_img)

                    frame_record = {
                        "frame_index": frame_count,
                        "host_frame_monotonic_ns": time.monotonic_ns(),
                        "external_camera_timestamp": float(
                            external_ts.camera_timestamp
                        ),
                        "wrist_camera_timestamp": float(wrist_ts.camera_timestamp),
                        "external_host_capture_monotonic_ns": int(
                            external_ts.host_capture_monotonic_ns
                        ),
                        "wrist_host_capture_monotonic_ns": int(
                            wrist_ts.host_capture_monotonic_ns
                        ),
                        "joint_position": joint_pos.tolist(),
                        "gripper_position": float(gripper_pos[0]),
                        "ee_position": np.asarray(ee_pos, dtype=np.float32).tolist(),
                        "ee_orientation_xyzw": np.asarray(
                            ee_quat_xyzw, dtype=np.float32
                        ).tolist(),
                    }

                    pending_frame = {
                        "external_img": external_img,
                        "wrist_img": wrist_img,
                        "blank": blank,
                        "joint_pos": joint_pos,
                        "gripper_pos": gripper_pos,
                        "frame_record": frame_record,
                        "action_gripper_state": gripper_state,
                    }

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
            _set_hold_control(ctrl)
            arm.panda.stop_controller()
        except Exception:
            pass

        if enable_logging and not reflex_error_occurred:
            try:
                _finalize_episode_data(
                    save_episode=frame_count > 0 or pending_frame is not None
                )
            except Exception as exc:
                print(f"\n[Error] Failed to finalize current episode: {exc}")
        elif enable_logging and reflex_error_occurred:
            try:
                _finalize_episode_data(save_episode=False)
            except Exception:
                pass
        if microphone_recorder is not None:
            try:
                microphone_recorder.stop()
            except Exception:
                pass
        try:
            reactive_desk_client.close()
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

        if enable_logging and dataset is not None:
            try:
                dataset.stop_image_writer()
            except Exception:
                pass

        if enable_logging and dataset_root is not None and not args.not_compute_vad:
            try:
                print("\n[VAD] Computing offline VAD metadata for collected audio...")
                compute_vad_for_dataset(
                    dataset_root,
                    overwrite=False,
                    save_clips=True,
                    print_summary=True,
                )
            except Exception as exc:
                print(f"[VAD] Warning: failed to compute VAD metadata: {exc}")


if __name__ == "__main__":
    main()
