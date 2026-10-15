from typing import Annotated

from fastapi import APIRouter, Depends

from replay.backend.dependencies import get_registry
from replay.backend.models import DatasetDetail, DatasetList, LatestEpisodeResponse
from replay.backend.registry import DatasetRegistry
from replay.backend.services.get_dataset import get_dataset
from replay.backend.services.get_latest_episode import get_latest_episode
from replay.backend.services.list_datasets import list_datasets

router = APIRouter(prefix="/datasets", tags=["datasets"])
Registry = Annotated[DatasetRegistry, Depends(get_registry)]


@router.get("", response_model=DatasetList)
def datasets(registry: Registry) -> DatasetList:
    return list_datasets(registry)


@router.get("/latest-episode", response_model=LatestEpisodeResponse)
def latest_episode(registry: Registry) -> LatestEpisodeResponse:
    return get_latest_episode(registry)


@router.get("/{dataset_id}", response_model=DatasetDetail)
def dataset(dataset_id: str, registry: Registry) -> DatasetDetail:
    return get_dataset(registry, dataset_id)
