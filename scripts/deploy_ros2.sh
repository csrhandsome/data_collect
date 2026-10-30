#!/usr/bin/env bash
# Inside the Docker shell: controllers, cameras, VR and warning-only diagnostics.
# Bringup unlocks the brakes and activates FCI through the standalone Desk client.
#
#   bash scripts/deploy_docker.sh
#   bash scripts/deploy_ros2.sh
#
# Second terminal, unchanged:
#   bash scripts/deploy_docker.sh
#   python -m vr_collect --config config/train/panda.yaml
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if ! command -v ros2 >/dev/null 2>&1; then
  echo 'ROS 2 environment missing. Run bash scripts/deploy_docker.sh first, then bash scripts/deploy_ros2.sh inside the container.' >&2
  exit 1
fi

cam_pid=""
vr_pid=""
monitor_pid=""
arm_pid=""
cleanup() {
  local status=$?
  trap - INT TERM EXIT
  for pid in "$arm_pid" "$vr_pid" "$cam_pid" "$monitor_pid"; do
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -- "-$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Own sessions so these launchers cannot signal the arm controllers.
setsid python -m control.ros2.diagnostics_bridge &
monitor_pid=$!
setsid python -m control.ros2.vr_node &
vr_pid=$!
setsid bash "$root/src/env/cameras.sh" &
cam_pid=$!

setsid bash "$root/src/env/bringup.sh" "$@" &
arm_pid=$!
wait "$arm_pid"
