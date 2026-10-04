"""Exercise embedded frame delivery, cache freshness and shared-shard isolation."""

import importlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from replay.backend.main import create_app
from replay.scripts.prepare_video import prepare_video

encoder = importlib.import_module("replay.scripts.prepare_video")


def encoded_image(color, format="PNG"):
    output = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(output, format=format)
    return output.getvalue()


@pytest.fixture(params=["v2.1", "v3.0"])
def image_dataset(tmp_path: Path, request):
    root = tmp_path / "images"
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": request.param,
        "fps": 20,
        "total_episodes": 2,
        "total_frames": 4,
        "features": {"camera": {"dtype": "image", "shape": [16, 16, 3]}},
    }
    (meta / "info.json").write_text(json.dumps(info))
    records = [{"episode_index": i, "length": 2, "tasks": []} for i in range(2)]
    data = [
        encoded_image("red"),
        encoded_image("green", "JPEG"),
        encoded_image("blue"),
        encoded_image("white"),
    ]
    rows = [
        {
            "camera": {"bytes": image, "path": None},
            "episode_index": i // 2,
            "frame_index": i % 2,
            "index": i,
            "timestamp": (i % 2) / 20,
        }
        for i, image in enumerate(data)
    ]
    paths = []
    if request.param == "v2.1":
        (meta / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in records))
        for i in range(2):
            path = root / f"data/chunk-000/episode_{i:06d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows[2 * i : 2 * i + 2]), path)
            paths.append(path)
    else:
        for i, record in enumerate(records):
            record.update(
                {
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": 2 * i,
                    "dataset_to_index": 2 * i + 2,
                }
            )
        episodes = meta / "episodes/chunk-000/file-000.parquet"
        episodes.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(records), episodes)
        path = root / "data/chunk-000/file-000.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
        paths = [path, path]
    return root, paths, data


def decoded(video):
    reader = imageio_ffmpeg.read_frames(str(video["path"]), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        return metadata, [
            np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3) for frame in reader
        ]
    finally:
        reader.close()


def test_real_h264_decode_range_and_episode_isolation(image_dataset):
    root, _, _ = image_dataset
    first = prepare_video(root, 0, "camera")
    second = prepare_video(root, 1, "camera")
    metadata, frames = decoded(first)
    assert metadata["fps"] == 20
    assert len(frames) == 2
    assert frames[0][0, 0, 0] > 220 and frames[0][0, 0, 2] < 30
    _, frames = decoded(second)
    assert frames[0][0, 0, 2] > 220 and frames[0][0, 0, 0] < 30
    assert first["path"] != second["path"]
    assert first["duration_s"] == pytest.approx(0.1)
    with TestClient(create_app(root)) as client:
        base = "/api/datasets/images/episodes/0"
        response = client.get(f"{base}/video?feature=camera")
        assert response.status_code == 200
        url = response.json()["url"]
        assert "version=" in url
        content = client.get(url, headers={"Range": "bytes=0-127"})
        assert content.status_code == 206
        assert content.headers["content-type"] == "video/mp4"
        assert content.content == first["path"].read_bytes()[:128]
        head = client.head(url)
        assert int(head.headers["content-length"]) == first["path"].stat().st_size
        assert client.get(url, headers={"Range": "bytes=999999999-"}).status_code == 416
        assert client.get(f"{base}/image?feature=camera").status_code == 404


def test_warm_cache_skips_parquet_and_encoding_and_refreshes(image_dataset, monkeypatch):
    root, paths, _ = image_dataset
    first = prepare_video(root, 0, "camera")
    with monkeypatch.context() as patch:
        patch.setattr(
            encoder, "_episode_table", lambda *a: pytest.fail("Warm cache read image column")
        )
        patch.setattr(encoder, "_encode", lambda *a: pytest.fail("Warm cache encoded again"))
        assert prepare_video(root, 0, "camera") == first
    table = pq.read_table(paths[0])
    rows = table.to_pylist()
    rows[0]["camera"]["bytes"] = encoded_image("yellow")
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), paths[0])
    fresh = prepare_video(root, 0, "camera")
    assert fresh["version"] != first["version"]
    assert not first["path"].exists()
    _, frames = decoded(fresh)
    assert frames[0][0, 0, 0] > 200 and frames[0][0, 0, 1] > 200


@pytest.mark.parametrize("operation", ["missing", "escape", "malformed"])
def test_warm_cache_does_not_hide_invalid_parquet(image_dataset, tmp_path, operation):
    root, paths, _ = image_dataset
    prepare_video(root, 0, "camera")
    path = paths[0]
    if operation == "missing":
        path.unlink()
    elif operation == "escape":
        outside = tmp_path / "outside.parquet"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        path.write_bytes(b"broken parquet")
    with TestClient(create_app(root)) as client:
        response = client.get("/api/datasets/images/episodes/0/video?feature=camera")
        assert response.status_code == (404 if operation == "missing" else 422)
        assert str(tmp_path) not in response.text


def test_external_image_path_is_rejected(image_dataset, tmp_path):
    root, paths, _ = image_dataset
    outside = tmp_path / "outside.png"
    outside.write_bytes(encoded_image("red"))
    table = pq.read_table(paths[0])
    rows = table.to_pylist()
    rows[0]["camera"] = {"bytes": None, "path": str(outside)}
    pq.write_table(pa.Table.from_pylist(rows), paths[0])
    with TestClient(create_app(root)) as client:
        response = client.get("/api/datasets/images/episodes/0/video?feature=camera")
        assert response.status_code == 422
        assert str(tmp_path) not in response.text
    assert not list(root.glob(".replay-cache/**/*.mp4"))


def test_path_images_and_odd_dimensions_invalidate_cache(image_dataset):
    root, paths, _ = image_dataset
    image = root / "camera.png"
    output = io.BytesIO()
    Image.new("RGB", (17, 15), "red").save(output, format="PNG")
    image.write_bytes(output.getvalue())
    table = pq.read_table(paths[0])
    rows = table.to_pylist()
    for row in rows:
        row["camera"] = {"bytes": None, "path": "camera.png"}
    pq.write_table(pa.Table.from_pylist(rows), paths[0])
    first = prepare_video(root, 0, "camera")
    metadata, _ = decoded(first)
    assert metadata["size"] == (18, 16)
    image.write_bytes(encoded_image("blue"))
    assert prepare_video(root, 0, "camera")["version"] != first["version"]


def test_changed_episode_boundaries_invalidate_cache(image_dataset):
    root, _, _ = image_dataset
    prepare_video(root, 0, "camera")
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] == "v2.1":
        path = root / "meta/episodes.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["length"], records[1]["length"] = 3, 1
        path.write_text("\n".join(json.dumps(record) for record in records))
    else:
        path = root / "meta/episodes/chunk-000/file-000.parquet"
        table = pq.read_table(path)
        records = table.to_pylist()
        records[0]["dataset_from_index"], records[0]["dataset_to_index"] = 2, 4
        records[1]["dataset_from_index"], records[1]["dataset_to_index"] = 0, 2
        pq.write_table(pa.Table.from_pylist(records, schema=table.schema), path)
    with TestClient(create_app(root)) as client:
        response = client.get("/api/datasets/images/episodes/0/video?feature=camera")
        assert response.status_code == 422


def test_irregular_host_timing_is_preserved(image_dataset):
    root, _, _ = image_dataset
    sidecar = root / "episode_000000.sync.json"

    def write_timing(second):
        sidecar.write_text(
            json.dumps(
                {
                    "episode_index": 0,
                    "frame_records": [
                        {"host_frame_monotonic_ns": 1000000000},
                        {"host_frame_monotonic_ns": 1000000000 + second},
                    ],
                }
            )
        )

    write_timing(200000000)
    first = prepare_video(root, 0, "camera")
    _, frames = decoded(first)
    assert first["duration_s"] == pytest.approx(0.25)
    assert len(frames) == 5
    assert frames[1][0, 0, 0] > 220
    assert frames[-1][0, 0, 1] > frames[-1][0, 0, 0]
    write_timing(300000000)
    assert prepare_video(root, 0, "camera")["duration_s"] == pytest.approx(0.35)
    write_timing(-10000000)
    with pytest.raises(ValueError, match="increasing"):
        prepare_video(root, 0, "camera")


def test_concurrent_requests_encode_once_and_retry_after_failure(image_dataset, monkeypatch):
    root, _, _ = image_dataset
    real_encode = encoder._encode
    calls = []

    def counted(*args):
        calls.append(1)
        return real_encode(*args)

    monkeypatch.setattr(encoder, "_encode", counted)
    with ThreadPoolExecutor(max_workers=4) as pool:
        videos = list(pool.map(lambda _: prepare_video(root, 0, "camera"), range(4)))
    assert len(calls) == 1
    assert all(video == videos[0] for video in videos)

    def failed(*args):
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(encoder, "_encode", failed)
    with pytest.raises(RuntimeError):
        prepare_video(root, 1, "camera")
    assert not list(root.glob(".replay-cache/**/encoding-*.mp4"))
    monkeypatch.setattr(encoder, "_encode", real_encode)
    assert prepare_video(root, 1, "camera")["path"].is_file()
