#!/usr/bin/env bash
# One terminal for the ROS environment: Panda controllers and both D435 streams.
# Desk FCI must already be on. This script does not activate FCI.
#
#   ./deploy_deocker.sh
#   bash src/env/start.sh
#
# Second terminal, unchanged:
#   ./deploy_deocker.sh
#   python -m vr_collect --config config/train/panda.yaml
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd /workspace

cam_pid=""
vr_pid=""
cleanup() {
  local status=$?
  trap - INT TERM EXIT
  for pid in "$vr_pid" "$cam_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  exit "$status"
}
trap cleanup INT TERM EXIT

# Own sessions so these launchers cannot signal the arm controllers.
setsid python -m control.ros2.vr_node &
vr_pid=$!
setsid bash "$root/cameras.sh" &
cam_pid=$!

bash "$root/bringup.sh" "$@"
