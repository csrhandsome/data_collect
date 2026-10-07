#!/usr/bin/env bash
set -euo pipefail
desktop_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$desktop_repo_root"
desktop_cpu_root="$desktop_repo_root/build/desktop/cpu"
desktop_python_project="$desktop_repo_root/replay/desktop/python"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$desktop_repo_root/build/desktop/uv-cache}"
export ELECTRON_BUILDER_CACHE="${ELECTRON_BUILDER_CACHE:-$desktop_repo_root/build/desktop/electron-builder-cache}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$desktop_cpu_root"

# Both environments are explicit and independent of the repository .venv.
UV_PROJECT_ENVIRONMENT="$desktop_cpu_root/runtime-venv" \
  uv sync --project "$desktop_python_project" --locked --no-default-groups --python 3.12
UV_PROJECT_ENVIRONMENT="$desktop_cpu_root/compiler-venv" \
  uv sync --project "$desktop_python_project" --locked --only-group packaging --python 3.12
pnpm --dir replay/frontend install --frozen-lockfile
pnpm --dir replay/desktop install --frozen-lockfile
pnpm --dir replay/desktop exec install-electron --no
pnpm --dir replay/frontend build
env -u PYTHONPATH "$desktop_cpu_root/compiler-venv/bin/python" \
  replay/desktop/scripts/build-runtime.py --build-dir "$desktop_cpu_root" \
  --dependency-python "$desktop_cpu_root/runtime-venv/bin/python" "$@"
pnpm --dir replay/desktop package
