"""Discover complete episodes published by the collection workflow."""

from replay.backend.models import LatestEpisode, LatestEpisodeResponse
from replay.backend.registry import DatasetRegistry
from replay.scripts.read_sidecars import read_sync


def get_latest_episode(registry: DatasetRegistry) -> LatestEpisodeResponse:
    latest = None
    for dataset_id, root in registry.entries().items():
        try:
            indices = set()
            for folder in [root, root / "audio"]:
                for path in folder.glob("episode_*.sync.json"):
                    number = path.name.removeprefix("episode_").removesuffix(".sync.json")
                    if number.isdigit():
                        indices.add(int(number))
            # Collection indices increase. Old sidecars without a publication
            # timestamp do not count as new collection events.
            for index in sorted(indices, reverse=True):
                sync = read_sync(root, index)
                stamp = sync.get("saved_at_ns")
                if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= 0:
                    continue
                candidate = LatestEpisode(
                    dataset_id=dataset_id, episode_index=index, saved_at_ns=str(stamp)
                )
                if latest is None or stamp > int(latest.saved_at_ns):
                    latest = candidate
                break
        except (OSError, ValueError, TypeError, RuntimeError):
            # Another dataset may be mid-save or deletion. Retry next poll,
            # keeping other completed datasets discoverable.
            continue
    return LatestEpisodeResponse(episode=latest)
