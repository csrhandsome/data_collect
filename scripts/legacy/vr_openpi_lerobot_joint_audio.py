"""Historical entry; all acquisition now uses the configured EE workflow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.legacy.launcher import launch

if __name__ == "__main__":
    launch()
