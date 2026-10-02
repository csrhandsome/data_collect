from replay.backend.models import DatasetList, DatasetSummary
from replay.backend.registry import DatasetRegistry
from replay.backend.services.get_dataset import get_dataset


def list_datasets(registry: DatasetRegistry) -> DatasetList:
    return DatasetList(
        datasets=[
            DatasetSummary.model_validate(get_dataset(registry, dataset_id).model_dump())
            for dataset_id in registry.entries()
        ]
    )
