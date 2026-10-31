#!/usr/bin/env python3
"""Trial-align downloaded tactile datasets to LeRobot v3. Paths are injected here."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data_analysis.data_clean import ALIGNERS  # noqa: E402

# Source trees on mirnagroup. Output stays on the local disk so the trial does not fill the NAS.
SOURCES = {
    "haptile": Path("/mnt/nas/HapTile"),
    "freetacman": Path("/mnt/nas/FreeTacMan"),
    "feelsight": Path("/mnt/nas/FeelSight"),
    "wirefishing": Path("/mnt/nas/WireFishing-M"),
    "touch_and_go": Path("/mnt/nas/Touch-and-Go"),
    "tactile_video_pretrain_nc": Path("/mnt/nas/tactile-video-pretrain-nc"),
    "vitamin": Path("/mnt/nas/ViTaMIn"),
}
OUTPUT_ROOT = Path("/home/mirnagroup/align-trial")
LIMIT = 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Align a few episodes of each tactile dataset to LeRobot v3")
    parser.add_argument("--dataset", choices=sorted(ALIGNERS), help="Only this dataset. Default: every injected path")
    parser.add_argument("--limit", type=int, default=LIMIT)
    args = parser.parse_args()
    names = [args.dataset] if args.dataset else list(SOURCES)
    failed = []
    for name in names:
        source = SOURCES[name]
        output = OUTPUT_ROOT / name
        print(f"[{name}] {source} -> {output}", flush=True)
        if output.exists():
            print(f"[{name}] 已有输出，跳过", flush=True)
            continue
        if not source.exists():
            print(f"[{name}] 源数据还没下到，跳过", flush=True)
            continue
        try:
            written = ALIGNERS[name]().align(source, output, limit=args.limit)
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"[{name}] 失败: {exc}", flush=True)
            traceback.print_exc()
            continue
        print(f"[{name}] 试转完成 {written}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
