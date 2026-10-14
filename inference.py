"""Run metadata-checked policy inference through the EE-only Panda controller."""

import argparse
import json
import logging
import signal
import threading
from pathlib import Path

from control.config import INFERENCE_CONFIG, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=INFERENCE_CONFIG)
    parser.add_argument(
        "--dry-run", action="store_true", help="Synthetic robot/cameras, real policy transport"
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--record", action="store_true", help="Save this inference run as an episode")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    from control.inference.loop import run_inference

    stop_event = threading.Event()
    previous = signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    try:
        stats = run_inference(
            load_config(args.config),
            dry_run=args.dry_run,
            max_steps=args.max_steps or (300 if args.dry_run else 0),
            record=args.record,
            stop_event=stop_event,
        )
        # uv can exit 130 after SIGINT even when its Python child exits cleanly.
        # Emit only after recording, return-to-start and resource cleanup finish.
        print(json.dumps({"event": "inference_finished", "episode": stats.get("episode")}), flush=True)
        return stats
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    main()
