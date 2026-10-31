#!/usr/bin/env bash
set -euo pipefail

replay_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$replay_repo_root"
export ROS2_HEALTH_PATH="${ROS2_HEALTH_PATH:-${ROS2_DATA_DIR:-$replay_repo_root/data}/runtime/ros2_health.json}"

for replay_tool in docker pnpm setsid; do
  if ! command -v "$replay_tool" >/dev/null 2>&1; then
    printf '缺少 %s，请先安装后再启动。\n' "$replay_tool" >&2
    exit 1
  fi
done

pnpm --dir src/replay/frontend install --frozen-lockfile
replay_runner_pid=''
replay_web_pid=''
replay_stop() {
  local status=$?
  trap - EXIT
  trap '' INT TERM
  for replay_pid in "$replay_runner_pid" "$replay_web_pid"; do
    if [[ -n "$replay_pid" ]]; then
      kill -s INT -- "-$replay_pid" 2>/dev/null || true
      wait "$replay_pid" 2>/dev/null || true
    fi
  done
  exit "$status"
}
trap replay_stop EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid env --default-signal=INT,TERM REPLAY_API_TARGET="${REPLAY_API_TARGET:-http://127.0.0.1:8003}" pnpm --dir src/replay/frontend dev --host 127.0.0.1 --port 5173 --strictPort &
replay_web_pid=$!
setsid env --default-signal=INT,TERM bash scripts/deploy_runner.sh --with-api &
replay_runner_pid=$!
printf '\n采集工作台：http://127.0.0.1:5173\nAPI 文档：http://127.0.0.1:8003/docs\n请先单独启动 deploy_ros2.sh；本入口启动监听 runner 和网页，Ctrl+C 保存并关闭它们。\n\n'
wait -n "$replay_runner_pid" "$replay_web_pid"
