#!/usr/bin/env bash
set -eo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
if ! command -v ros2 >/dev/null 2>&1; then
  echo 'ROS 2 environment missing. Run bash scripts/deploy_docker.sh first.' >&2
  exit 1
fi
python src/env/check_controller.py
/usr/bin/python3 src/env/check_launch.py --robot-config-file "$root/src/env/franka.config.yaml"
exec python src/env/desk_bringup.py \
  robot_config_file:="$root/src/env/franka.config.yaml" "$@"
