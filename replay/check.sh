#!/usr/bin/env bash
set -euo pipefail
replay_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$replay_repo_root"
uv sync
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest replay/tests/test_readers.py -q
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest replay/tests/test_api.py replay/tests/test_images.py -q
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest replay/tests/test_operations.py tests/test_episode_deleter.py -q
env -u PYTHONPATH uv run ruff check replay
pnpm --dir replay/frontend install --frozen-lockfile
pnpm --dir replay/frontend build
