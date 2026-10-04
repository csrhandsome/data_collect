#!/usr/bin/env bash
set -euo pipefail

replay_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$replay_repo_root"

for replay_tool in uv pnpm setsid; do
  if ! command -v "$replay_tool" >/dev/null 2>&1; then
    printf '缺少 %s，请先安装后再启动。\n' "$replay_tool" >&2
    exit 1
  fi
done

uv sync
pnpm --dir replay/frontend install --frozen-lockfile
replay_api_pid=''
replay_web_pid=''
replay_stop() {
  trap - EXIT INT TERM
  for replay_pid in "$replay_api_pid" "$replay_web_pid"; do
    if [[ -n "$replay_pid" ]]; then
      kill -- "-$replay_pid" 2>/dev/null || true
      wait "$replay_pid" 2>/dev/null || true
    fi
  done
}
trap replay_stop EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid env -u PYTHONPATH uv run replay-server &
replay_api_pid=$!
setsid env REPLAY_API_TARGET="${REPLAY_API_TARGET:-http://127.0.0.1:8003}" pnpm --dir replay/frontend dev --host 127.0.0.1 --port 5173 --strictPort &
replay_web_pid=$!
printf '\n回放页面：http://127.0.0.1:5173\nAPI 文档：http://127.0.0.1:8003/docs\nCtrl+C 关闭前后端。\n\n'
wait -n "$replay_api_pid" "$replay_web_pid"
