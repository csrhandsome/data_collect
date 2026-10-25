"""Merge local LeRobot v3 MP4 datasets, preserving audio and trajectory sidecars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_data_root() -> Path:
    return _repo_root() / "data" / "dataset"


def _is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _resolve_dataset(name_or_path: str, data_root: Path) -> Path:
    raw = Path(name_or_path).expanduser()
    candidates = []
    if raw.is_absolute() or raw.parts[:1] in ((".",), ("..",)):
        candidates.append(raw)
    else:
        candidates.append(data_root / raw)
        if raw.parts[:1] == ("openpi",):
            candidates.append(data_root / raw.name)
        candidates.append(raw)

    for candidate in candidates:
        resolved = candidate.resolve()
        if _is_lerobot_dataset(resolved):
            return resolved

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"找不到有效 LeRobot 数据集: {name_or_path} (tried: {tried})")


def _default_output_name(inputs: list[str]) -> str:
    names = [Path(item).name for item in inputs]
    if len(names) == 1:
        return f"{names[0]}_merged"
    if len(names) == 2:
        return f"{names[0]}__plus__{names[1]}"
    return f"{names[0]}__plus_{len(names) - 1}_datasets"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="合并本地 data/dataset 下的 LeRobot 数据集")
    parser.add_argument(
        "datasets",
        nargs="+",
        help="输入数据集名或路径，例如 franka_lerobot_6_19_audio_v3_mp4",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="输出数据集名；默认使用 <第一个>__plus__<第二个>",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_default_data_root(),
        help="包含数据集目录的根目录，默认 data/dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出目录已存在，先删除再重新生成",
    )
    args = parser.parse_args(argv)

    if len(args.datasets) < 2:
        print("[ERROR] 至少传入两个数据集名。", file=sys.stderr)
        return 2

    try:
        merge_datasets(
            args.datasets,
            output_name=args.output,
            data_root=args.data_root.resolve(),
            overwrite=bool(args.overwrite),
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


def merge_datasets(
    input_names: list[str], *, output_name: str | None, data_root: Path, overwrite: bool
) -> Path:
    from data_analysis.merge_v3 import merge_v3
    from replay.scripts._common import _load_metadata

    paths = [_resolve_dataset(name, data_root) for name in input_names]
    for path in paths:
        _load_metadata(path)
    return merge_v3(
        paths, data_root / (output_name or _default_output_name(input_names)), overwrite
    )


if __name__ == "__main__":
    raise SystemExit(main())
