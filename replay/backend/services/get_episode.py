from replay.backend.data_access import parse_dataset, validated
from replay.backend.models import EpisodeDetail
from replay.backend.registry import DatasetRegistry


def get_episode(registry: DatasetRegistry, dataset_id: str, episode_index: int) -> EpisodeDetail:
    from replay.scripts.read_episode import read_episode

    result = parse_dataset(read_episode, registry.get(dataset_id), episode_index)
    return validated(EpisodeDetail, {**result, "dataset_id": dataset_id})
