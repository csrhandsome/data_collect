#!/usr/bin/env bash
set -euo pipefail
desktop_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$desktop_repo_root"
export ELECTRON_BUILDER_CACHE="${ELECTRON_BUILDER_CACHE:-$desktop_repo_root/build/desktop/electron-builder-cache}"
uv sync --locked --group packaging
pnpm --dir replay/frontend install --frozen-lockfile
pnpm --dir replay/desktop install --frozen-lockfile
pnpm --dir replay/desktop exec install-electron --no
pnpm --dir replay/frontend build
env -u PYTHONPATH uv run --locked --group packaging python replay/desktop/scripts/build-runtime.py
pnpm --dir replay/desktop package
