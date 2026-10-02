"""Single EE collection entry point; configure joint/EE labels in panda.yaml."""

import argparse
import logging
from pathlib import Path

from control.config import DEFAULT_CONFIG, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run", action="store_true", help="Synthetic robot, VR, cameras and audio"
    )
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    from control.collection.pipeline import run_collection

    return run_collection(
        config, dry_run=args.dry_run, max_steps=args.max_steps or (300 if args.dry_run else 0)
    )


if __name__ == "__main__":
    main()
