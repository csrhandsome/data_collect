"""Root-project entry point: uv run replay-server."""

import argparse
import os

from control.config import DEFAULT_CONFIG, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = load_config(args.config).get("replay", {})
    os.environ.setdefault("REPLAY_DATA_ROOT", config.get("data_root", "data/openpi"))
    import uvicorn

    uvicorn.run(
        "replay.backend.main:app",
        host=args.host or config.get("host", "127.0.0.1"),
        port=args.port or int(config.get("port", 8003)),
    )
