from replay.backend.data_access import parse_dataset, validated
from replay.backend.models import DatasetDetail
from replay.backend.registry import DatasetRegistry


def get_dataset(registry: DatasetRegistry, dataset_id: str) -> DatasetDetail:
    from replay.scripts.read_dataset import read_dataset

    result = parse_dataset(read_dataset, registry.get(dataset_id))
    return validated(DatasetDetail, {**result, "id": dataset_id})
