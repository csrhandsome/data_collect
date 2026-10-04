"""Shared parser boundary: validate results and sanitize parsing failures."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from replay.backend.errors import ReplayError
from replay.backend.registry import DatasetRegistry
from replay.scripts._common import MissingDatasetFile

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def parse_dataset(parser: Callable[..., dict], root: Path, *args: Any) -> dict:
    try:
        return parser(root, *args)
    except ReplayError:
        raise
    except KeyError as exc:
        raise ReplayError(404, "Episode or feature was not found.") from exc
    except (MissingDatasetFile, FileNotFoundError, NotADirectoryError) as exc:
        raise ReplayError(404, "Dataset file was not found.") from exc
    except ValueError as exc:
        raise ReplayError(422, "Dataset is malformed or the feature is unsupported.") from exc
    except (OSError, TypeError, IndexError, OverflowError, RuntimeError) as exc:
        raise ReplayError(422, "Dataset could not be read. Check its files and metadata.") from exc


def validated(model: type[ResponseModel], payload: dict) -> ResponseModel:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise ReplayError(422, "Dataset metadata does not match the supported schema.") from exc


def video_source(
    registry: DatasetRegistry, dataset_id: str, episode_index: int, feature: str
) -> dict:
    from replay.scripts.resolve_video import resolve_video

    root = registry.get(dataset_id)
    source = parse_dataset(resolve_video, root, episode_index, feature)
    try:
        path = Path(source["path"]).resolve()
    except (TypeError, ValueError, KeyError, OSError, RuntimeError) as exc:
        raise ReplayError(422, "Video metadata is malformed.") from exc
    if not path.is_relative_to(root):
        raise ReplayError(422, "Video path must remain inside its dataset.")
    if not path.is_file():
        raise ReplayError(404, "Video file was not found.")
    return {**source, "path": path}
