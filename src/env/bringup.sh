#!/usr/bin/env bash
set -eo pipefail
cd /workspace
python src/env/check_controller.py
exec ros2 launch panda_cartesian_controller panda_bringup.launch.py \
  robot_config_file:=/workspace/src/env/franka.config.yaml "$@"
