#!/usr/bin/env python3
"""
Collect Franka teleop data into a LeRobot dataset using DROID-style keys.

- Cameras: external + wrist (2x RealSense)
- Control: VR controller pose via teleop_xr + IntegratedVelocity + DROID IK solver
- Actions: normalized joint velocity [-1, 1] + gripper state
- Output: LeRobot dataset (no intermediate HDF5)

Usage example:

uv run vr_openpi_lerobot_franka.py \
  --instruction "Pick up the wide-mouth bottle" \
  --external-camera-serial 825412070292 \
  --wrist-camera-serial 825412070487 \
  --color-only
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from transforms3d.quaternions import qinverse, qmult, quat2axangle

from droid_ik_solver import DroidIKSolver
from realsense_connector import RealSenseConnector
from control.robotic_arm_controller import RoboticArmControler
from control.robotic_arm_controller import _LatestFrameBuffer
from control.robotic_arm_controller import _camera_capture_worker


def _get_local_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


@dataclass
class ControllerPose:
    position: np.ndarray
    orientation_wxyz: np.ndarray


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _get_controller_device(state: Any, handedness: str) -> Any | None:
    for device in getattr(state, "devices", []):
        role = _enum_value(getattr(device, "role", ""))
        hand = _enum_value(getattr(device, "handedness", ""))
        if role == "controller" and hand == handedness:
            return device
    return None


def _get_controller_pose(device: Any) -> ControllerPose | None:
    if device is None:
        return None

    pose = getattr(device, "gripPose", None) or getattr(device, "pose", None)
    if pose is None:
        return None

    pos = getattr(pose, "position", None)
    ori = getattr(pose, "orientation", None)
    if pos is None or ori is None:
        return None

    position = np.array(
        [
            float(pos.get("x", 0.0)),
            float(pos.get("y", 0.0)),
            float(pos.get("z", 0.0)),
        ],
        dtype=np.float64,
    )
    quat = np.array(
        [
            float(ori.get("w", 1.0)),
            float(ori.get("x", 0.0)),
            float(ori.get("y", 0.0)),
            float(ori.get("z", 0.0)),
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        quat = quat / norm

    return ControllerPose(position=position, orientation_wxyz=quat)


def _button_pressed(device: Any, index: int) -> bool:
    if device is None:
        return False
    gamepad = getattr(device, "gamepad", None)
    if gamepad is None:
        return False
    buttons = getattr(gamepad, "buttons", [])
    if index < 0 or index >= len(buttons):
        return False
    return bool(getattr(buttons[index], "pressed", False))


class TeleopXRInput:
    """Background teleop_xr server + latest XR state cache."""

    def __init__(self, host: str, port: int, input_mode: str = "controller") -> None:
        try:
            from teleop_xr import Teleop
            from teleop_xr.config import TeleopSettings, InputMode
            from teleop_xr.messages import XRState
        except Exception as exc:
            raise RuntimeError(
                "teleop_xr is required for VR control. Install it first, e.g.\n"
                "  uv add 'teleop-xr[ik]@git+https://github.com/qrafty-ai/teleop_xr'"
            ) from exc

        if input_mode == "controller":
            mode_enum = InputMode.CONTROLLER
        elif input_mode == "hand":
            mode_enum = InputMode.HAND
        else:
            mode_enum = InputMode.AUTO

        settings = TeleopSettings(host=host, port=port, input_mode=mode_enum)
        self._teleop = Teleop(settings=settings)
        self._xr_state_type = XRState

        self._state_lock = threading.Lock()
        self._latest_state = None
        self._last_update_s: float = 0.0
        self._thread: threading.Thread | None = None

        def _on_xr_update(_pose: np.ndarray, message: dict[str, Any]) -> None:
            try:
                payload = message.get("data", message)
                state = self._xr_state_type.model_validate(payload)
            except Exception:
                return

            with self._state_lock:
                self._latest_state = state
                self._last_update_s = time.time()

        self._teleop.subscribe(_on_xr_update)

    def connect(self) -> None:
        self._thread = threading.Thread(target=self._teleop.run, daemon=True)
        self._thread.start()

    def wait_for_first_state(self, timeout_s: float) -> bool:
        start = time.time()
        while True:
            if self.get_latest_state() is not None:
                return True
            if timeout_s > 0 and (time.time() - start) > timeout_s:
                return False
            time.sleep(0.01)

    def get_latest_state(self) -> Any | None:
        with self._state_lock:
            return self._latest_state

    def last_update_age_s(self) -> float:
        with self._state_lock:
            if self._last_update_s <= 0:
                return float("inf")
            return max(0.0, time.time() - self._last_update_s)

    def disconnect(self) -> None:
        try:
            self._teleop.stop()
        except Exception:
            pass


class VRToDroidMapper:
    """Map teleop_xr controller motion to DROID cartesian velocity [-1, 1]."""

    def __init__(
        self,
        *,
        max_lin_step: float,
        max_rot_step: float,
        deadman_both_squeeze: bool,
    ) -> None:
        self.max_lin_step = max(float(max_lin_step), 1e-6)
        self.max_rot_step = max(float(max_rot_step), 1e-6)
        self.deadman_both_squeeze = bool(deadman_both_squeeze)

        self._prev_right_pose: ControllerPose | None = None

    def reset(self) -> None:
        self._prev_right_pose = None

    def compute(self, xr_state: Any) -> tuple[np.ndarray, dict[str, bool], bool]:
        left = _get_controller_device(xr_state, "left")
        right = _get_controller_device(xr_state, "right")

        left_squeeze = _button_pressed(left, 1)
        right_squeeze = _button_pressed(right, 1)

        if self.deadman_both_squeeze:
            deadman = left_squeeze and right_squeeze
        else:
            deadman = right_squeeze

        right_pose = _get_controller_pose(right)
        control_active = deadman and right_pose is not None

        # Quest-like mapping (xr-standard gamepad)
        # index 4: primary (A/X), index 5: secondary (B/Y)
        right_primary = _button_pressed(right, 4)
        right_secondary = _button_pressed(right, 5)
        left_primary = _button_pressed(left, 4)
        left_secondary = _button_pressed(left, 5)

        buttons = {
            "gripper_open": right_primary,
            "gripper_close": right_secondary,
            "episode_toggle": left_primary or left_secondary,
        }

        cart_vel = np.zeros(6, dtype=np.float64)

        if not control_active:
            self._prev_right_pose = None
            return cart_vel, buttons, False

        assert right_pose is not None

        if self._prev_right_pose is None:
            self._prev_right_pose = right_pose
            return cart_vel, buttons, True

        pos_prev = self._prev_right_pose.position
        quat_prev = self._prev_right_pose.orientation_wxyz
        pos_now = right_pose.position
        quat_now = right_pose.orientation_wxyz

        delta_pos = pos_now - pos_prev
        dq = qmult(quat_now, qinverse(quat_prev))
        axis, angle = quat2axangle(dq)
        if abs(angle) < 1e-12 or np.any(np.isnan(axis)):
            rot_vec = np.zeros(3, dtype=np.float64)
        else:
            if angle > np.pi:
                angle -= 2.0 * np.pi
            rot_vec = np.asarray(axis, dtype=np.float64) * float(angle)

        cart_vel[:3] = delta_pos / self.max_lin_step
        cart_vel[3:] = rot_vec / self.max_rot_step
        cart_vel = np.clip(cart_vel, -1.0, 1.0)

        self._prev_right_pose = right_pose
        return cart_vel, buttons, True


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
                "Fix the dataset first (e.g. delete/prune incomplete data), then retry."
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
        description="Collect Franka data into LeRobot with teleop_xr VR input"
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
            "VR sensitivity multiplier (0-1]. "
            "1.0 = full DROID speed (0.075 m/step translation, 0.15 rad/step rotation)."
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

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4443)
    parser.add_argument(
        "--input-mode",
        type=str,
        default="controller",
        choices=["controller", "hand", "auto"],
    )
    parser.add_argument(
        "--xr-startup-timeout-s",
        type=float,
        default=60.0,
        help="Seconds to wait for first XR state after server starts.",
    )
    parser.add_argument(
        "--vr-max-lin-step",
        type=float,
        default=0.075,
        help="Meters of controller translation per control step mapped to cart_vel=1.0",
    )
    parser.add_argument(
        "--vr-max-rot-step",
        type=float,
        default=0.15,
        help="Radians of controller rotation per control step mapped to cart_vel=1.0",
    )
    parser.add_argument(
        "--deadman-both-squeeze",
        action="store_true",
        default=True,
        help="Require both controllers squeeze to move (default: on).",
    )
    parser.add_argument(
        "--deadman-right-only",
        action="store_true",
        help="Use only right squeeze as deadman switch.",
    )

    args = parser.parse_args()

    if args.control_frequency <= 0:
        raise ValueError("--control-frequency must be > 0")

    enable_logging = not args.no_logging
    if enable_logging and not args.instruction.strip():
        raise ValueError("--instruction is required when logging is enabled")

    deadman_both_squeeze = bool(args.deadman_both_squeeze and not args.deadman_right_only)

    print("=" * 70)
    print("Franka LeRobot data collection (VR + joint velocity actions)")
    print("=" * 70)
    print(f"Control frequency: {args.control_frequency} Hz")
    print(f"Sensitivity: {args.sensitivity} (1.0 = full DROID speed)")
    print(f"Action epsilon: {args.action_epsilon}")
    print(
        "Camera stream: "
        f"{args.camera_width}x{args.camera_height}@{args.camera_fps} "
        f"(depth={'off' if args.color_only else 'on'})"
    )
    print(f"Input device: teleop_xr WebXR ({args.input_mode})")
    print(f"Teleop host/port: {args.host}:{args.port}")
    print(f"External camera serial: {args.external_camera_serial}")
    print(f"Wrist camera serial: {args.wrist_camera_serial}")
    if enable_logging:
        date = "2_24"
        args.repo_id = f"{args.repo_id}_{date}"
        print(f"Logging: enabled (max {args.max_duration} s)")
        print(f"Instruction: {args.instruction}")
        print(f"LeRobot repo_id: {args.repo_id}")
    else:
        print("Logging: disabled")
    print("=" * 70)

    print("Starting teleop_xr server...")
    vr_input = TeleopXRInput(host=args.host, port=args.port, input_mode=args.input_mode)
    vr_input.connect()

    local_ip = _get_local_ip()
    print("Open this URL in your headset/browser:")
    print(f"  https://{local_ip}:{args.port}")
    print("Wait for VR connection and enter XR mode...")

    if not vr_input.wait_for_first_state(timeout_s=float(args.xr_startup_timeout_s)):
        raise RuntimeError(
            "Timed out waiting for XR state. Ensure headset opened the URL and entered XR mode."
        )
    print("[VR] First XR state received.")

    mapper = VRToDroidMapper(
        max_lin_step=float(args.vr_max_lin_step),
        max_rot_step=float(args.vr_max_rot_step),
        deadman_both_squeeze=deadman_both_squeeze,
    )

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
        if args.camera_startup_timeout_s > 0 and elapsed > args.camera_startup_timeout_s:
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
    last_episode_toggle_pressed = False
    last_vr_drop_warning_s = 0.0

    print("\nControl mapping:")
    print("  Hold squeeze to enable motion")
    if deadman_both_squeeze:
        print("    deadman: LEFT squeeze + RIGHT squeeze")
    else:
        print("    deadman: RIGHT squeeze")
    print("  Right controller pose: 6DoF motion")
    print("  Right A(primary): gripper open | Right B(secondary): gripper close")
    print("  Left X/Y(primary/secondary): save episode + reset")
    print("    press again with no frames to quit")

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

                xr_state = vr_input.get_latest_state()
                if xr_state is None:
                    time.sleep(0.001)
                    continue

                state_age_s = vr_input.last_update_age_s()
                if state_age_s > 0.5:
                    now = time.time()
                    if now - last_vr_drop_warning_s > 1.0:
                        print(
                            f"[VR] No fresh XR updates for {state_age_s:.2f}s; holding still...",
                            end="\r",
                        )
                        last_vr_drop_warning_s = now

                cart_vel, vr_buttons, control_active = mapper.compute(xr_state)

                if vr_buttons["episode_toggle"] and not last_episode_toggle_pressed:
                    if enable_logging and frame_count > 0:
                        print("\n[Control] Episode toggle pressed, saving and resetting...")
                        ctrl.set_control(np.zeros(7))
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
                        finally:
                            frame_count = 0
                            recording_started = False
                            recording_started_at = None

                        arm.panda.stop_controller()
                        arm.gripper_open()
                        gripper_state = 1.0
                        last_gripper_cmd = 1.0
                        mapper.reset()
                        arm.move_to_start()

                        ctrl = controllers.IntegratedVelocity()
                        arm.panda.start_controller(ctrl)
                        ctrl.set_control(np.zeros(7))
                    else:
                        print("\n[Control] Episode toggle pressed, stopping...")
                        break

                last_episode_toggle_pressed = vr_buttons["episode_toggle"]

                sens = float(args.sensitivity)
                cart_vel = cart_vel * sens

                if args.action_epsilon > 0.0:
                    small_mask = np.abs(cart_vel) <= args.action_epsilon
                    if np.any(small_mask):
                        cart_vel[small_mask] = 0.0

                if not control_active:
                    cart_vel[:] = 0.0

                robot_state = arm.panda.get_state()
                qpos = np.asarray(robot_state.q, dtype=np.float64)
                qvel = np.asarray(robot_state.dq, dtype=np.float64)
                joint_delta, normalized_action = ik_solver.solve(cart_vel, qpos, qvel)

                if control_active and not gripper_busy:
                    ctrl.set_control(joint_delta)
                elif not gripper_busy:
                    ctrl.set_control(np.zeros(7))

                gripper_changed = False
                gripper_cmd = last_gripper_cmd
                if vr_buttons["gripper_open"]:
                    gripper_cmd = 1.0
                elif vr_buttons["gripper_close"]:
                    gripper_cmd = 0.0

                now = time.time()
                if now - last_gripper_switch_time < gripper_switch_cooldown_s:
                    gripper_cmd = last_gripper_cmd

                if gripper_cmd != last_gripper_cmd and not gripper_busy:
                    gripper_changed = True
                    last_gripper_switch_time = now
                    gripper_state = 1.0 if gripper_cmd > 0.5 else 0.0
                    last_gripper_cmd = gripper_cmd
                    gripper_busy = True

                    def _do_gripper(cmd: float) -> None:
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

                    threading.Thread(target=_do_gripper, args=(gripper_cmd,), daemon=True).start()

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
                    gripper_pos = np.asarray([np.float32(gripper_state)], dtype=np.float32)
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
            vr_input.disconnect()
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
