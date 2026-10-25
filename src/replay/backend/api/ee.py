from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from replay.backend.dependencies import get_registry
from replay.backend.models import EETrajectory
from replay.backend.registry import DatasetRegistry
from replay.backend.services.get_ee import get_ee

router = APIRouter(prefix="/datasets/{dataset_id}/episodes/{episode_index}", tags=["ee"])


@router.get("/ee", response_model=EETrajectory)
def ee(
    dataset_id: str,
    episode_index: Annotated[int, Path(ge=0)],
    registry: Annotated[DatasetRegistry, Depends(get_registry)],
    feature: Annotated[str, Query(min_length=1)] = "ee_pose",
    max_points: Annotated[int, Query(ge=2, le=20000)] = 2000,
) -> EETrajectory:
    return get_ee(registry, dataset_id, episode_index, feature, max_points)
