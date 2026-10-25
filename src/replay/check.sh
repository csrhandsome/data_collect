#!/usr/bin/env bash
set -euo pipefail
replay_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$replay_repo_root"
uv sync
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest src/replay/tests/test_readers.py -q
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest src/replay/tests/test_api.py src/replay/tests/test_images.py -q
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest src/replay/tests/test_operations.py scripts/tests/test_episode_deleter.py scripts/tests/test_lerobot_upgrade.py -q
env -u PYTHONPATH uv run ruff check src/replay
pnpm --dir src/replay/frontend install --frozen-lockfile
pnpm --dir src/replay/frontend build
