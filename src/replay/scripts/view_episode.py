"""Read-only CLI for historical acquisition datasets; the web UI synchronizes media."""

import argparse
import json
from pathlib import Path

from replay.backend.config import configured_data_root
from replay.scripts.read_audio import read_audio
from replay.scripts.read_dataset import read_dataset
from replay.scripts.read_episode import read_episode
from replay.scripts.read_series import read_series


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--episode", "--episode-index", type=int, default=0)
    parser.add_argument("--list-episodes", action="store_true")
    parser.add_argument("--max-points", type=int, default=20)
    args = parser.parse_args(argv)
    data_root = configured_data_root()
    if args.root:
        root = args.root
    elif args.repo_id:
        root = data_root / args.repo_id.rsplit("/", 1)[-1]
    else:
        candidates = [p.parent.parent for p in data_root.glob("*/meta/info.json")]
        if not candidates:
            parser.error(f"No datasets found in {data_root}")
        root = max(candidates, key=lambda p: (p / "meta/info.json").stat().st_mtime_ns)
    dataset = read_dataset(root)
    if args.list_episodes:
        print(json.dumps(dataset["episodes"], ensure_ascii=False, indent=2))
        return dataset["episodes"]
    episode = read_episode(root, args.episode)
    print(
        json.dumps(
            {"root": str(root), "fps": dataset["fps"], **episode}, ensure_ascii=False, indent=2
        )
    )
    for block in episode["blocks"]:
        if block["kind"] in {"series", "ee"}:
            values = read_series(root, args.episode, block["key"], args.max_points)
            print(block["key"], values["total_points"], "samples", values["bounds"])
        elif block["kind"] == "audio":
            audio = read_audio(root, args.episode)
            print("audio", audio["sample_rate"], "Hz", audio["duration_s"], "s")
    return episode


if __name__ == "__main__":
    main()
