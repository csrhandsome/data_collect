"""Save a human annotation without rewriting the recorded trajectory."""

from replay.backend.data_access import parse_dataset
from replay.backend.errors import ReplayError
from replay.backend.models import EpisodeDetail
from replay.backend.registry import DatasetRegistry
from replay.backend.services.get_episode import get_episode
from replay.scripts.read_sidecars import update_sync


def annotate_episode(
    registry: DatasetRegistry,
    dataset_id: str,
    episode_index: int,
    success: bool | None,
    expected_saved_at_ns: str | None = None,
) -> EpisodeDetail:
    detail = get_episode(registry, dataset_id, episode_index)
    if expected_saved_at_ns is not None and detail.saved_at_ns != expected_saved_at_ns:
        raise ReplayError(409, "片段已变化，请重新扫描数据后再标注。")
    parse_dataset(update_sync, registry.get(dataset_id), episode_index, {"success": success})
    return detail.model_copy(update={"success": success})
