"""Record measured native Panda state. This diagnostic does not send motion targets."""

import argparse
import csv
from pathlib import Path

from control.config import DEFAULT_CONFIG, load_config
from control.robotic_arm_controller import RoboticArmControler
from control.util.timing import FixedRate


def record_robot_trace(output, *, config_path=DEFAULT_CONFIG, dry_run=False, samples=200):
    from control._panda.fake import FakeBackend

    if samples <= 0:
        raise ValueError("samples must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["timestamp"]
            + [f"q{i}" for i in range(7)]
            + [f"transform{i}" for i in range(16)]
            + [f"command{i}" for i in range(7)]
        )
        with RoboticArmControler(
            config=load_config(config_path), backend=FakeBackend() if dry_run else None
        ) as arm:
            initial = arm.wait_ready()
            rate = FixedRate(100)
            for _ in range(samples):
                rate.tick()
                state = arm.get_state()
                writer.writerow(
                    [(state.sampled_monotonic_ns - initial.sampled_monotonic_ns) / 1e9]
                    + state.joint_positions.tolist()
                    + state.end_effector_pose.flatten(order="F").tolist()
                    + initial.joint_positions.tolist()
                )
    print(f"Saved {samples} measured samples: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    record_robot_trace(
        args.output, config_path=args.config, dry_run=args.dry_run, samples=args.samples
    )


if __name__ == "__main__":
    main()
