#!/usr/bin/env bash
# 启动唯一的 Franka ROS 2 镜像并进入 shell。在仓库根目录执行。
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

image="data-collect/franka-ros2-jazzy:dev"
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "missing image ${image}; build src/env/Dockerfile first" >&2
  exit 1
fi

exec docker run --rm -it \
  --network host \
  --ipc host \
  --cap-add SYS_NICE \
  --ulimit rtprio=99 \
  --ulimit memlock=-1 \
  --device-cgroup-rule "c 13:* rwm" \
  --device-cgroup-rule "c 81:* rwm" \
  --device-cgroup-rule "c 189:* rwm" \
  --device-cgroup-rule "c 188:* rwm" \
  --device-cgroup-rule "c 166:* rwm" \
  --device-cgroup-rule "c 116:* rwm" \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" \
  -e FRANKA_USERNAME \
  -e FRANKA_PASSWORD \
  -e ROS2_DESK_AUTO_UNLOCK \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e DATA_COLLECT_CONFIG=/workspace/config/train/panda.yaml \
  -e DATA_COLLECT_INFERENCE_CONFIG=/workspace/config/inference/panda.yaml \
  -e SDL_VIDEODRIVER=dummy \
  -e SDL_AUDIODRIVER=dummy \
  -e ROS2_HEALTH_PATH=/workspace/data/runtime/ros2_health.json \
  -v "$root:/workspace" \
  -v /dev:/dev \
  -v "${ROS2_DATA_DIR:-$root/data}:/workspace/data" \
  -w /workspace \
  "$image" \
  bash "$@"
