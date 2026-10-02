"""Read-only embedded images, audio and numeric streams."""

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, Response

from replay.backend.data_access import parse_dataset
from replay.backend.dependencies import get_registry
from replay.scripts.read_audio import read_audio, resolve_audio
from replay.scripts.read_image import read_image, read_image_metadata
from replay.scripts.read_series import read_series

router = APIRouter(prefix="/datasets/{dataset_id}/episodes/{episode_index}")


@router.get("/image")
def image(
    dataset_id: str,
    episode_index: int,
    feature: str,
    frame_index: int = Query(0, ge=0),
    registry=Depends(get_registry),
):
    data = parse_dataset(read_image, registry.get(dataset_id), episode_index, feature, frame_index)
    return Response(data, media_type="image/png")


@router.get("/series")
def series(
    dataset_id: str,
    episode_index: int,
    feature: str,
    max_points: int = Query(2000, ge=2, le=10000),
    registry=Depends(get_registry),
):
    return parse_dataset(read_series, registry.get(dataset_id), episode_index, feature, max_points)


@router.get("/audio")
def audio(dataset_id: str, episode_index: int, registry=Depends(get_registry)):
    return parse_dataset(read_audio, registry.get(dataset_id), episode_index)


@router.get("/audio/file")
def audio_file(dataset_id: str, episode_index: int, registry=Depends(get_registry)):
    path, _ = parse_dataset(resolve_audio, registry.get(dataset_id), episode_index)
    return FileResponse(path, media_type="audio/wav")


@router.get("/image/metadata")
def image_metadata(
    dataset_id: str, episode_index: int, feature: str, registry=Depends(get_registry)
):
    return parse_dataset(read_image_metadata, registry.get(dataset_id), episode_index, feature)
