from urllib.parse import quote, urlencode

from replay.backend.data_access import validated, video_source
from replay.backend.models import VideoMetadata
from replay.backend.registry import DatasetRegistry


def get_video(
    registry: DatasetRegistry, dataset_id: str, episode_index: int, feature: str
) -> VideoMetadata:
    source = video_source(registry, dataset_id, episode_index, feature)
    path = f"/api/datasets/{quote(dataset_id, safe='')}/episodes/{episode_index}/video/content"
    return validated(
        VideoMetadata,
        {
            **source,
            "dataset_id": dataset_id,
            "episode_index": episode_index,
            "url": f"{path}?{urlencode({'feature': feature})}",
        },
    )
