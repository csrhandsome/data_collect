#!/usr/bin/env python3
"""Download one of the NAS datasets. This is the only download entry."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data_analysis.data_download.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
