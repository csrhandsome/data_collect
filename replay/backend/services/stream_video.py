"""Serve a confined MP4 with single-range seeking for browser playback."""

import re
from collections.abc import Iterator
from typing import BinaryIO

from starlette.background import BackgroundTask
from starlette.responses import Response, StreamingResponse

from replay.backend.data_access import video_source
from replay.backend.errors import ReplayError
from replay.backend.registry import DatasetRegistry

_RANGE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK_SIZE = 1024 * 256


def _byte_range(value: str, size: int) -> tuple[int, int] | None:
    match = _RANGE.fullmatch(value.strip())
    if match is None or size == 0:
        return None
    left, right = match.groups()
    if not left and not right:
        return None
    try:
        if not left:
            suffix = int(right)
            return (max(0, size - suffix), size - 1) if suffix > 0 else None
        start = int(left)
        end = min(int(right), size - 1) if right else size - 1
    except ValueError:
        return None
    if start >= size or end < start:
        return None
    return start, end


def _chunks(file: BinaryIO, remaining: int) -> Iterator[bytes]:
    try:
        while remaining:
            chunk = file.read(min(_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
    finally:
        file.close()


def stream_video(
    registry: DatasetRegistry,
    dataset_id: str,
    episode_index: int,
    feature: str,
    range_header: str | None = None,
    head: bool = False,
) -> Response:
    source = video_source(registry, dataset_id, episode_index, feature)
    try:
        size = source["path"].stat().st_size
    except FileNotFoundError as exc:
        raise ReplayError(404, "Video file was not found.") from exc
    except OSError as exc:
        raise ReplayError(422, "Video file could not be read.") from exc
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
    start, end, status = 0, size - 1, 200
    if range_header is not None:
        selected = _byte_range(range_header, size)
        if selected is None:
            headers["Content-Range"] = f"bytes */{size}"
            return Response(status_code=416, headers=headers)
        start, end = selected
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    length = max(0, end - start + 1)
    headers["Content-Length"] = str(length)
    if head:
        return Response(status_code=status, media_type="video/mp4", headers=headers)
    try:
        file = source["path"].open("rb")
        file.seek(start)
    except FileNotFoundError as exc:
        raise ReplayError(404, "Video file was not found.") from exc
    except OSError as exc:
        raise ReplayError(422, "Video file could not be read.") from exc
    return StreamingResponse(
        _chunks(file, length),
        status_code=status,
        media_type="video/mp4",
        headers=headers,
        background=BackgroundTask(file.close),
    )
