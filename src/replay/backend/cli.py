"""Root-project entry point: uv run replay-server."""

import argparse
import os
from pathlib import Path

from control.config import DEFAULT_CONFIG, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--start-collect", action="store_true", help="Start one managed collection runner with the API")
    args = parser.parse_args()
    os.environ["DATA_COLLECT_CONFIG"] = str(Path(args.config).resolve())
    config = load_config(args.config).get("replay", {})
    os.environ.setdefault("REPLAY_DATA_ROOT", config.get("data_root", "data/dataset"))
    import uvicorn

    from replay.backend.main import create_app
    from replay.backend.operations import OperationManager

    application = create_app(
        operation_manager=OperationManager(
            config_path=Path(args.config).resolve(),
            pythonpath=os.environ.get("PYTHONPATH"),
        ),
        start_collect=args.start_collect,
        shutdown_timeout=None,
    )

    uvicorn.run(
        application,
        host=args.host or config.get("host", "127.0.0.1"),
        port=args.port or int(config.get("port", 8003)),
    )


if __name__ == "__main__":
    main()
