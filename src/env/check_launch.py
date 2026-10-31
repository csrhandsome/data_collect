"""Evaluate the installed launch actions without starting nodes or hardware."""

import argparse
import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext


def check(robot_config_file):
    launch_file = (
        Path(get_package_share_directory("panda_cartesian_controller"))
        / "launch/panda_bringup.launch.py"
    )
    spec = importlib.util.spec_from_file_location("panda_bringup_preflight", launch_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate_launch_description()
    context = LaunchContext()
    context.launch_configurations["robot_config_file"] = str(robot_config_file)
    actions = module.spawn(context)
    if not actions:
        raise RuntimeError("Panda launch generated no actions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config-file", required=True)
    args = parser.parse_args()
    check(args.robot_config_file)
    print("Launch preflight passed; no nodes or hardware started.")
