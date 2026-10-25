#!/usr/bin/env bash
set -eo pipefail
cd /workspace
python src/env/check_controller.py
exec ros2 launch franka_bringup example.launch.py \
  controller_names:=cartesian_impedance_example_controller \
  robot_config_file:=/workspace/src/env/franka.config.yaml "$@"
