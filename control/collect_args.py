import argparse


def _build_parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description)


def _add_repo_and_instruction_args(
    parser: argparse.ArgumentParser,
    *,
    repo_id_default: str,
    repo_id_help: str | None = None,
    include_date: bool = False,
    date_default: str = "3_10",
    include_second_instruction: bool = False,
) -> None:
    repo_kwargs: dict[str, object] = {
        "type": str,
        "default": repo_id_default,
    }
    if repo_id_help is not None:
        repo_kwargs["help"] = repo_id_help
    parser.add_argument("--repo-id", **repo_kwargs)
    if include_date:
        parser.add_argument("--date", type=str, default=date_default)
    parser.add_argument("--instruction", type=str, default="")
    if include_second_instruction:
        parser.add_argument("--second-instruction", type=str, default="")


def _add_droid_control_args(
    parser: argparse.ArgumentParser,
    *,
    control_frequency_default: float,
    sensitivity_default: float,
    sensitivity_help: str,
) -> None:
    parser.add_argument(
        "--control-frequency", type=float, default=control_frequency_default
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=sensitivity_default,
        help=sensitivity_help,
    )
    parser.add_argument("--action-epsilon", type=float, default=1e-6)


def _add_joint_control_args(
    parser: argparse.ArgumentParser,
    *,
    control_frequency_default: float,
    sensitivity_default: float,
    sensitivity_help: str,
) -> None:
    parser.add_argument(
        "--control-frequency", type=float, default=control_frequency_default
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=sensitivity_default,
        help=sensitivity_help,
    )
    parser.add_argument(
        "--max-ee-translation-step",
        type=float,
        default=0.04,
        help="Maximum EE translation increment per control cycle (m/step) at full VR input.",
    )
    parser.add_argument(
        "--max-ee-rotation-step",
        type=float,
        default=0.04,
        help="Maximum EE rotation increment per control cycle (rad/step) at full VR input.",
    )
    parser.add_argument("--action-epsilon", type=float, default=1e-6)


def _add_camera_and_logging_args(
    parser: argparse.ArgumentParser,
    *,
    color_only_help: str | None = None,
    startup_timeout_help: str | None = None,
) -> None:
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    color_kwargs: dict[str, object] = {"action": "store_true"}
    if color_only_help is not None:
        color_kwargs["help"] = color_only_help
    parser.add_argument("--color-only", **color_kwargs)
    startup_kwargs: dict[str, object] = {"type": float, "default": 10.0}
    if startup_timeout_help is not None:
        startup_kwargs["help"] = startup_timeout_help
    parser.add_argument("--camera-startup-timeout-s", **startup_kwargs)
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument("--max-duration", type=float, default=3600.0)
    parser.add_argument("--external-camera-serial", type=str, default=None)
    parser.add_argument("--wrist-camera-serial", type=str, default=None)
    parser.add_argument("--image-hw", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, default=0.9)
    parser.add_argument("--no-logging", action="store_true")


def _add_vr_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vr-host", type=str, default="0.0.0.0")
    parser.add_argument("--vr-port", type=int, default=4443)
    parser.add_argument(
        "--vr-long-press-s",
        type=float,
        default=0.5,
        help="Seconds both triggers must be held to enable arm movement.",
    )


def _add_vr_droid_mapping_args(parser: argparse.ArgumentParser) -> None:
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
        help="VR rotation (rad) from neutral that maps to full speed. 0.35 ~= 20 deg.",
    )
    parser.add_argument(
        "--vr-enable-rotation",
        action="store_true",
        help="Enable rotation control from VR wrist. Off by default to avoid IK chaos.",
    )


def _add_vr_joint_mapping_args(parser: argparse.ArgumentParser) -> None:
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


def _add_audio_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-channels", type=int, default=1)


def _add_soft_gripper_args(parser: argparse.ArgumentParser) -> None:
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


def build_vr_lerobot_droid_parser() -> argparse.ArgumentParser:
    parser = _build_parser("Collect Franka data into LeRobot via VR (teleop_xr)")
    _add_repo_and_instruction_args(
        parser,
        repo_id_default="openpi/franka_droid_lerobot",
    )
    _add_droid_control_args(
        parser,
        control_frequency_default=15.0,
        sensitivity_default=1.0,
        sensitivity_help="Cartesian velocity multiplier (0-1]. 1.0 = full DROID speed.",
    )
    _add_camera_and_logging_args(parser)
    _add_vr_connection_args(parser)
    _add_vr_droid_mapping_args(parser)
    return parser


def build_vr_lerobot_joint_parser() -> argparse.ArgumentParser:
    parser = _build_parser("Collect Franka data into LeRobot via VR (teleop_xr)")
    _add_repo_and_instruction_args(
        parser,
        repo_id_default="openpi/franka_franka_lerobot",
        include_date=True,
        date_default="3_10",
    )
    _add_joint_control_args(
        parser,
        control_frequency_default=20.0,
        sensitivity_default=0.9,
        sensitivity_help="Global multiplier on the per-step EE command increments.",
    )
    _add_camera_and_logging_args(parser)
    _add_vr_connection_args(parser)
    _add_vr_joint_mapping_args(parser)
    return parser


def build_vr_lerobot_joint_two_prompt_parser() -> argparse.ArgumentParser:
    parser = _build_parser("Collect Franka data into LeRobot via VR (teleop_xr)")
    _add_repo_and_instruction_args(
        parser,
        repo_id_default="openpi/franka_franka_lerobot",
        include_date=True,
        date_default="3_10",
        include_second_instruction=True,
    )
    _add_joint_control_args(
        parser,
        control_frequency_default=20.0,
        sensitivity_default=0.9,
        sensitivity_help="Global multiplier on the per-step EE command increments.",
    )
    _add_camera_and_logging_args(parser)
    _add_vr_connection_args(parser)
    _add_vr_joint_mapping_args(parser)
    return parser


def build_vr_lerobot_joint_audio_parser() -> argparse.ArgumentParser:
    parser = _build_parser(
        "Collect Franka data into LeRobot via VR (teleop_xr) with audio"
    )
    _add_repo_and_instruction_args(
        parser,
        repo_id_default="openpi/franka_franka_lerobot",
        include_date=True,
        date_default="3_10",
    )
    _add_joint_control_args(
        parser,
        control_frequency_default=20.0,
        sensitivity_default=0.9,
        sensitivity_help="Global multiplier on the per-step EE command increments.",
    )
    _add_camera_and_logging_args(parser)
    _add_audio_args(parser)
    _add_vr_connection_args(parser)
    _add_vr_joint_mapping_args(parser)
    return parser


def build_vr_lerobot_joint_force_parser() -> argparse.ArgumentParser:
    parser = _build_parser("Collect Franka data into LeRobot via VR (teleop_xr)")
    _add_repo_and_instruction_args(
        parser,
        repo_id_default="openpi/franka_franka_lerobot",
        include_second_instruction=True,
    )
    _add_joint_control_args(
        parser,
        control_frequency_default=30.0,
        sensitivity_default=0.9,
        sensitivity_help="Global multiplier on the per-step EE command increments.",
    )
    _add_camera_and_logging_args(parser)
    _add_soft_gripper_args(parser)
    _add_vr_connection_args(parser)
    _add_vr_joint_mapping_args(parser)
    return parser


def build_xbox_lerobot_droid_parser() -> argparse.ArgumentParser:
    parser = _build_parser("Collect Franka data into LeRobot (DROID-style keys)")
    _add_repo_and_instruction_args(
        parser,
        repo_id_default="openpi/franka_droid_lerobot",
        repo_id_help="Repo id for storing all episodes (all prompts share the same repo).",
        include_second_instruction=True,
    )
    _add_droid_control_args(
        parser,
        control_frequency_default=15.0,
        sensitivity_default=1.0,
        sensitivity_help=(
            "Joystick sensitivity multiplier (0-1]. "
            "1.0 = full DROID speed (0.075 m/step translation, 0.15 rad/step rotation). "
            "0.5 = half speed. Applied before IK solver."
        ),
    )
    _add_camera_and_logging_args(
        parser,
        color_only_help="Disable depth stream to reduce USB bandwidth.",
        startup_timeout_help="Seconds to wait for first frames (0 to wait indefinitely).",
    )
    return parser
