#!/usr/bin/env bash
# Only subscribe to the existing VR topic through the ROS2 CLI.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
exec python3 -u "$root/scripts/record_vr_stick.py" "$@"
