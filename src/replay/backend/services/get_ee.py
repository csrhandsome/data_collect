from replay.backend.data_access import parse_dataset, validated
from replay.backend.models import EETrajectory
from replay.backend.registry import DatasetRegistry


def get_ee(
    registry: DatasetRegistry,
    dataset_id: str,
    episode_index: int,
    feature: str,
    max_points: int,
) -> EETrajectory:
    from replay.scripts.read_ee import read_ee

    result = parse_dataset(read_ee, registry.get(dataset_id), episode_index, feature, max_points)
    return validated(
        EETrajectory,
        {**result, "dataset_id": dataset_id, "episode_index": episode_index},
    )
