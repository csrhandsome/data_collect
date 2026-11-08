"""One command for every NAS download."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_analysis.data_download.catalog import DOWNLOADS, find_download
from data_analysis.data_download.download_tactile import Settings
from data_analysis.data_download.runner import run_download


def list_lines() -> list[str]:
    lines = []
    for item in DOWNLOADS:
        batch = f" --batch {item.batch}" if item.batch else ""
        lines.append(f"{item.name}{batch}\t{item.folder}\t{item.kind}\t{item.detail}")
    return lines


def main(argv: list[str] | None = None) -> None:
    names = sorted({item.name for item in DOWNLOADS})
    parser = argparse.ArgumentParser(description="下载一份数据集到本机暂存，再传到 NAS")
    parser.add_argument("--list", action="store_true", help="列出数据集和来源")
    parser.add_argument("--dataset", choices=names, help="只下载这一份")
    parser.add_argument(
        "--batch",
        default="",
        help="egocentric_100k 的批次: original、additional、extra_500gb",
    )
    parser.add_argument("--manifest", type=Path, help="egocentric_100k 的 selection-manifest.json")
    parser.add_argument("--stage", type=Path, default=Path("/home/mirnagroup/tactile-stage"))
    parser.add_argument("--nas", type=Path, default=Path("/mnt/nas"))
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(list_lines()))
        return
    if not args.dataset:
        parser.error("需要 --dataset，或用 --list 查看")
    spec = find_download(args.dataset, args.batch)
    if spec.kind == "manifest" and args.manifest is None:
        parser.error(f"{spec.name} --batch {spec.batch} 需要 --manifest")
    run_download(spec, Settings(stage=args.stage, nas=args.nas), args.manifest)


if __name__ == "__main__":
    main()
