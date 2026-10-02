from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Request, Response

from replay.backend.dependencies import get_registry
from replay.backend.models import VideoMetadata
from replay.backend.registry import DatasetRegistry
from replay.backend.services.get_video import get_video
from replay.backend.services.stream_video import stream_video

router = APIRouter(prefix="/datasets/{dataset_id}/episodes/{episode_index}", tags=["video"])
Registry = Annotated[DatasetRegistry, Depends(get_registry)]
EpisodeIndex = Annotated[int, Path(ge=0)]
VideoFeature = Annotated[str, Query(min_length=1)]


@router.get("/video", response_model=VideoMetadata)
def video(
    dataset_id: str,
    episode_index: EpisodeIndex,
    registry: Registry,
    feature: VideoFeature = "exterior_image_1_left",
) -> VideoMetadata:
    return get_video(registry, dataset_id, episode_index, feature)


@router.head("/video/content", include_in_schema=False)
@router.get("/video/content", response_class=Response)
def video_content(
    request: Request,
    dataset_id: str,
    episode_index: EpisodeIndex,
    registry: Registry,
    feature: VideoFeature = "exterior_image_1_left",
) -> Response:
    return stream_video(
        registry,
        dataset_id,
        episode_index,
        feature,
        range_header=request.headers.get("range"),
        head=request.method == "HEAD",
    )
