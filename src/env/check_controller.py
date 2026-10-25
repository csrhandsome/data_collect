"""Read-only check for a known upstream demo loop that overwrites external targets."""

import re
from pathlib import Path


def check(source):
    text = Path(source).read_text()
    update = re.search(
        r"CartesianImpedanceExampleController::update\([\s\S]*?(?=\n(?:void|Eigen::))", text
    )
    if update and re.search(r"^\s*updateMotionTarget\(", update.group(), re.MULTILINE):
        raise RuntimeError(
            "Official controller calls updateMotionTarget() unconditionally inside update(); "
            "its autonomous demo overwrites equilibrium_pose. Use an upstream revision that "
            "supports exclusively external targets before launching real hardware. "
            "No controller source has been modified."
        )
    if '"~/equilibrium_pose"' not in text:
        raise RuntimeError("Controller source does not expose ~/equilibrium_pose")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        default=(
            "/opt/franka_ws/src/franka_ros2/franka_example_controllers/src/fr3/"
            "cartesian_impedance_example_controller.cpp"
        ),
    )
    args = parser.parse_args()
    try:
        check(args.source)
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"Controller preflight failed: {exc}\n")
    print("Controller preflight passed; verify target tracking on your robot before collection.")
