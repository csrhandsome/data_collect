#!/usr/bin/env bash
# Subscribe to an independently started ROS deployment; never start or stop drivers.
# deploy_page.sh uses --with-api so the page can manage the same collector.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '用法：bash scripts/deploy_runner.sh [--max-steps N]\n仅启动监听 ROS2 话题的采集 runner；请先单独运行 deploy_ros2.sh。\n需要网页和 runner 时改用 bash scripts/deploy_page.sh；不要同时运行两个入口。\nCtrl+C 结束采集并保存；独立启动的 ROS2 驱动继续运行。\n'
  exit 0
fi

inside=false
with_api=false
mode_args=()
while [[ "${1:-}" == --inside || "${1:-}" == --with-api ]]; do
  case "$1" in
    --inside) inside=true ;;
    --with-api) with_api=true; mode_args=(--with-api) ;;
  esac
  shift
done
if [[ -f /.dockerenv ]] && command -v ros2 >/dev/null 2>&1; then
  inside=true
fi
if (( $# > 0 )) && { [[ $# != 2 || "$1" != --max-steps || ! "$2" =~ ^[1-9][0-9]*$ ]]; }; then
  echo '仅支持 --max-steps 正整数；运行配置使用 config/train/panda.yaml。' >&2
  exit 2
fi

if [[ "$inside" == false ]]; then
  if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
    echo '请先安装并启动 Docker Engine 与 Docker Compose。' >&2
    exit 1
  fi
  if ! docker image inspect data-collect/franka-ros2-jazzy:dev >/dev/null 2>&1; then
    echo '缺少 ROS2 镜像，请先运行 docker compose -f src/env/compose.yaml build。' >&2
    exit 1
  fi
  printf '启动 ROS2 采集监听 runner；驱动、相机和 VR 需由 deploy_ros2.sh 单独启动。\n'
  exec docker compose -f "$root/src/env/compose.yaml" run --rm --no-deps -T \
    --name data-collect-runner \
    ros2 bash scripts/deploy_runner.sh --inside "${mode_args[@]}" "$@"
fi

if ! command -v ros2 >/dev/null 2>&1; then
  echo '容器内缺少 ROS2 环境；请在宿主机直接运行 bash scripts/deploy_runner.sh。' >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
python - <<'PY'
from control.config import load_config

config = load_config()
if config['robot']['backend'] != 'ros2':
    raise SystemExit('deploy_runner.sh 要求 config/train/panda.yaml 的 robot.backend 为 ros2。')
PY

collect_pid=''
cleanup() {
  local status=$?
  trap - EXIT
  trap '' INT TERM
  if [[ -n "$collect_pid" ]]; then
    # SIGINT lets the collector finish recording before its container exits.
    kill -s INT -- "-$collect_pid" 2>/dev/null || true
    wait "$collect_pid" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '\n连接已有 ROS2 话题，等待头显连接和手柄操作。\nCtrl+C 只结束 runner；机械臂、相机和 VR 服务继续运行。\n\n'
if [[ "$with_api" == true ]]; then
  runtime=(python -m replay.backend.cli --start-collect --host 127.0.0.1 --port 8003)
else
  runtime=(python -m vr_collect --config config/train/panda.yaml "$@")
fi
setsid env --default-signal=INT,TERM "${runtime[@]}" &
collect_pid=$!
wait "$collect_pid"
