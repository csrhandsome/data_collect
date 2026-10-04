"""Backend configuration, kept separate from dataset parsing."""

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data/dataset"


def configured_data_root(data_root: Path | None = None) -> Path:
    return Path(data_root or os.environ.get("REPLAY_DATA_ROOT", DEFAULT_DATA_ROOT)).resolve()
