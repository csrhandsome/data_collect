"""Translate UR YAML defaults into the official launch file's arguments."""

import os
import sys
from pathlib import Path

from control.yaml_config import load_yaml_config


def launch_command(config, overrides=()):
    robot, cfg = config["robot"], config["ros2"]
    if robot.get("backend") != "ur_ros2":
        raise ValueError("UR bringup requires robot.backend=ur_ros2")
    arguments = {
        "ur_type": robot.get("type", "ur5"),
        "robot_ip": robot["hostname"],
        "launch_rviz": "false",
        "headless_mode": "false",
        "initial_joint_controller": cfg.get("controller_name", "scaled_joint_trajectory_controller"),
        "use_mock_hardware": str(cfg.get("use_mock_hardware", False)).lower(),
    }
    if cfg.get("kinematics_params_file"):
        arguments["kinematics_params_file"] = cfg["kinematics_params_file"]
    # Overrides are ROS launch name:=value arguments; never evaluated by a shell.
    for value in overrides:
        key, sep, arg = value.partition(":=")
        if not sep or not key or not arg:
            raise ValueError(f"Expected ROS launch argument name:=value, got {value!r}")
        arguments[key] = arg
    return ["ros2", "launch", "ur_robot_driver", "ur_control.launch.py",
            *(f"{key}:={value}" for key, value in arguments.items())]


def main():
    default = Path(__file__).resolve().parents[2] / "config/train/ur.yaml"
    config = load_yaml_config(Path(os.environ.get("DATA_COLLECT_CONFIG", default)))
    command = launch_command(config, sys.argv[1:])
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
