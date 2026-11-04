#!/usr/bin/env bash
# On the host or inside Docker: controllers, cameras, VR and diagnostics.
# Bringup unlocks the brakes and activates FCI through the standalone Desk client.
#
#   bash scripts/deploy_ros2.sh
#
# Second terminal, unchanged:
#   bash scripts/deploy_runner.sh
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

inside=false
if [[ "${1:-}" == --inside ]]; then
  inside=true
  shift
fi
if [[ -f /.dockerenv || -f /run/.containerenv ]] || \
    grep -Eq '(docker|containerd|kubepods|libpod)' /proc/1/cgroup 2>/dev/null; then
  inside=true
fi

if [[ "$inside" == false ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo 'Docker is missing; deploy_ros2.sh uses deploy_docker.sh to start the ROS 2 container. Install and start Docker first.' >&2
    exit 1
  fi
  printf 'Host environment detected; starting ROS 2 inside Docker.\n'
  exec bash "$root/scripts/deploy_docker.sh" scripts/deploy_ros2.sh --inside "$@"
fi

if ! command -v ros2 >/dev/null 2>&1; then
  echo 'ROS 2 environment missing inside the container. Use the project image via bash scripts/deploy_docker.sh, or load its ROS 2 environment first.' >&2
  exit 1
fi
printf 'Container environment detected; starting ROS 2 directly.\n'

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
