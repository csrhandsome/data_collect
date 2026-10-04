"""Move to the configured start joints, then release robot control."""

import argparse
import logging

import numpy as np

from control.config import DEFAULT_CONFIG, load_config
from control.robotic_arm_controller import RoboticArmControler
from control.util.robot import start_joint_position


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    target = start_joint_position(config)
    with RoboticArmControler(config=config) as arm:
        result = arm.move_to_start()
        error = np.max(np.abs(result.final_state.joint_positions - target))
        print(f"Motion: {result.status}; max joint error: {error:.6f} rad", flush=True)
        print(f"Measured joints: {result.final_state.joint_positions.tolist()}", flush=True)
    print("Robot control released.", flush=True)


if __name__ == "__main__":
    main()
