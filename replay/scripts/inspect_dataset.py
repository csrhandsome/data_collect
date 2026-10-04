"""Inspect dataset metadata, EE samples, and video offsets without starting the API.

From repository root: uv run python -m replay.scripts.inspect_dataset
replay/demo_data/demo_v30 --episode 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .read_dataset import read_dataset
from .read_ee import read_ee
from .read_episode import read_episode
from .resolve_video import resolve_video


def inspect_dataset(root: Path, episode_index: int | None = None) -> dict:
    """Return the catalog, optionally enriched with an episode's visualization previews."""
    dataset = read_dataset(root)
    if episode_index is None:
        return dataset
    episode = read_episode(root, episode_index)
    previews = {}
    for block in episode["blocks"]:
        if block["kind"] == "ee":
            previews[block["key"]] = read_ee(root, episode_index, block["key"], max_points=8)
        elif block["kind"] == "video":
            video = resolve_video(root, episode_index, block["key"])
            previews[block["key"]] = {**video, "path": str(video["path"])}
    return {"dataset": dataset, "episode": episode, "previews": previews}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dataset root containing meta/info.json")
    parser.add_argument("--episode", type=int, help="Episode index to inspect")
    arguments = parser.parse_args()
    try:
        print(
            json.dumps(
                inspect_dataset(arguments.dataset, arguments.episode), indent=2, ensure_ascii=False
            )
        )
    except (ValueError, KeyError) as error:
        parser.exit(1, f"Cannot inspect dataset: {error}\n")
