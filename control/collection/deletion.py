"""The public API and command-line adapter for saved episode deletion."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EpisodeDeletionResult:
    dataset_dir: Path
    episode_index: int
    deleted_frames: int
    total_episodes_before: int
    total_episodes_after: int
    total_frames_before: int
    total_frames_after: int
    dry_run: bool


class EpisodeDeleter:
    """Delete saved v2 episodes and renumber subsequent episodes atomically.

    Returns a result on success; invalid inputs and transaction failures raise
    exceptions. Close recording resources before deletion and reopen the dataset
    afterwards. A full dataset copy on the same filesystem is required.
    """

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()

    def preview_episode(self, episode_index: int) -> EpisodeDeletionResult:
        """Validate and return the planned change without writing any files."""
        return self.delete_episode(episode_index, dry_run=True)

    def delete_episode(self, episode_index: int, *, dry_run: bool = False) -> EpisodeDeletionResult:
        """Validate a full copy, update references, then exchange directories.

        Preserves ASR annotations, updates VAD references, holds a deletion lock
        and checks the source for concurrent changes before committing.
        """
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise TypeError("episode_index must be an integer")
        if episode_index < 0:
            raise ValueError("episode_index must be >= 0")
        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be a boolean")

        from control.collection._episode_deletion import _delete_episode

        return _delete_episode(self.dataset_dir, episode_index, dry_run=dry_run)


def cli(argv: list[str] | None = None) -> int:
    """The single deletion command; exit codes belong only to this adapter."""
    parser = argparse.ArgumentParser(description="删除 LeRobot v2 episode 并重排后续编号")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--yes", action="store_true", help="执行删除；默认只预览")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        result = EpisodeDeleter(args.dataset).delete_episode(
            args.episode_index, dry_run=not args.yes
        )
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[INTERRUPTED] 删除操作被中断。", file=sys.stderr)
        return 130

    print(f"Dataset: {result.dataset_dir}")
    print(f"Delete episode_index: {result.episode_index}")
    print(f"Deleted frames: {result.deleted_frames}")
    print(f"Total episodes: {result.total_episodes_before} -> {result.total_episodes_after}")
    print(f"Total frames: {result.total_frames_before} -> {result.total_frames_after}")
    if result.episode_index < result.total_episodes_before - 1:
        first, last = result.episode_index + 1, result.total_episodes_before - 1
        print(f"Renumber episodes: {first:06d}->{first - 1:06d} ... {last:06d}->{last - 1:06d}")
    else:
        print("Renumber episodes: none (deleting latest episode)")
    print("[DRY-RUN] 未修改任何文件。加 --yes 执行删除。" if result.dry_run else "[OK] 删除完成。")
    return 0
