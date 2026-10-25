from typing import Annotated

from fastapi import APIRouter, Depends, Path

from replay.backend.api.operations import Manager
from replay.backend.dependencies import get_registry
from replay.backend.models import EpisodeAnnotation, EpisodeDetail, EpisodeList
from replay.backend.registry import DatasetRegistry
from replay.backend.services.annotate_episode import annotate_episode
from replay.backend.services.get_dataset import get_dataset
from replay.backend.services.get_episode import get_episode

router = APIRouter(prefix="/datasets/{dataset_id}/episodes", tags=["episodes"])
Registry = Annotated[DatasetRegistry, Depends(get_registry)]
EpisodeIndex = Annotated[int, Path(ge=0)]


@router.get("", response_model=EpisodeList)
def episodes(dataset_id: str, registry: Registry) -> EpisodeList:
    return EpisodeList(episodes=get_dataset(registry, dataset_id).episodes)


@router.get("/{episode_index}", response_model=EpisodeDetail)
def episode(dataset_id: str, episode_index: EpisodeIndex, registry: Registry) -> EpisodeDetail:
    return get_episode(registry, dataset_id, episode_index)


@router.post("/{episode_index}/annotation", response_model=EpisodeDetail)
def annotation(
    dataset_id: str,
    episode_index: EpisodeIndex,
    payload: EpisodeAnnotation,
    registry: Registry,
    manager: Manager,
) -> EpisodeDetail:
    with manager.annotation_access():
        return annotate_episode(
            registry, dataset_id, episode_index, payload.success, payload.expected_saved_at_ns
        )
