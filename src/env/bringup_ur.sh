#!/usr/bin/env bash
# Official UR ROS2 driver only. The user starts this independently of the runner.
set -eo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
exec /usr/bin/python3 src/env/bringup_ur.py "$@"
