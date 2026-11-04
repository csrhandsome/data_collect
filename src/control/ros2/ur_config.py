"""Validation for the experimental, six-axis UR ROS2 transport."""

from control.util.robot import start_joint_position
from control.util.timing import positive_rate

UR_JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


def validate_ur_config(config):
    robot, cfg = config.get("robot", {}), config.get("ros2", {})
    if robot.get("activate_fci", False):
        raise ValueError("UR does not support activate_fci")
    if config.get("gripper", {}).get("type", "none") not in ("none", "dh5"):
        raise ValueError("UR supports gripper.type=none or dh5; Franka gripper is unavailable")
    start_joint_position(config)
    names = cfg.get("joint_names", list(UR_JOINT_NAMES))
    if (not isinstance(names, list) or len(names) != 6
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != 6):
        raise ValueError("ros2.joint_names must contain six distinct nonempty names")
    for key in ("base_frame", "tool_frame", "controller_name"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"ros2.{key} must be a nonempty string")
    for key in ("trajectory_topic", "joint_state_topic", "robot_description_topic"):
        if key in cfg and (not isinstance(cfg[key], str) or not cfg[key].strip()):
            raise ValueError(f"ros2.{key} must be a nonempty string")
    for key, default in (("state_timeout_s", 0.25), ("connect_timeout_s", 10),
                         ("command_duration_s", 0.15), ("max_joint_velocity_rad_s", 0.3),
                         ("max_command_joint_error_rad", 0.15)):
        positive_rate(cfg.get(key, default), f"ros2.{key}")
    if float(cfg.get("command_duration_s", 0.15)) > 0.5:
        raise ValueError("UR command_duration_s must not exceed 0.5 seconds")
    if float(cfg.get("max_command_joint_error_rad", 0.15)) > 0.2:
        raise ValueError("UR max_command_joint_error_rad must not exceed 0.2")
    if not isinstance(cfg.get("use_mock_hardware", False), bool):
        raise ValueError("ros2.use_mock_hardware must be a boolean")
