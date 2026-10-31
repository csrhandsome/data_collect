"""Read-only check that the Panda controller accepts external targets only."""

import re
from pathlib import Path


def check(source):
    text = Path(source).read_text()
    update = re.search(r"::update\([\s\S]*?(?=\n\S)", text)
    if update and re.search(r"updateMotionTarget\s*\(", update.group()):
        raise RuntimeError(
            "Controller calls updateMotionTarget() inside update(); "
            "its autonomous demo overwrites equilibrium_pose."
        )
    if update and re.search(r"\b(?:std::)?(?:sin|cos)\s*\(", update.group()):
        raise RuntimeError(
            "Controller update() contains a periodic demo trajectory. "
            "External equilibrium_pose targets must be the only motion source."
        )
    if '"~/equilibrium_pose"' not in text:
        raise RuntimeError("Controller source does not expose ~/equilibrium_pose")


def check_installed(library):
    path = Path(library)
    if not path.is_file():
        raise RuntimeError(
            f"Panda Cartesian controller library is not installed: {path}. Rebuild the image."
        )
    if b"filter_coefficient" not in path.read_bytes():
        raise RuntimeError(
            "Installed controller predates configurable target filtering. "
            "Rebuild the image and recreate this container before starting ROS."
        )
    if b"~/joint_reset" not in path.read_bytes():
        raise RuntimeError(
            "Installed controller predates joint reset support. "
            "Rebuild the image and recreate this container before starting ROS."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        default=(
            "/workspace/src/env/panda_cartesian_controller/src/"
            "cartesian_pose_impedance_controller.cpp"
        ),
    )
    parser.add_argument(
        "--library",
        default=(
            "/opt/franka_ws/install/panda_cartesian_controller/lib/"
            "libpanda_cartesian_controller.so"
        ),
    )
    parser.add_argument("--skip-library", action="store_true")
    args = parser.parse_args()
    try:
        check(args.source)
        if not args.skip_library:
            check_installed(args.library)
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"Controller preflight failed: {exc}\n")
    print("Controller preflight passed; verify target tracking on the Panda before collection.")
