"""Exercise HTTP behavior against actual generated v2.1/v3.0 files."""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from replay.backend.main import create_app


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from replay.scripts.generate_demo import generate_demo

    root = tmp_path_factory.mktemp("api-demo")
    generate_demo(root)
    return root


@pytest.fixture(scope="module")
def client(demo_root: Path):
    with TestClient(create_app(demo_root)) as connection:
        yield connection


def _copy_dataset(source: Path, destination: Path, name: str = "demo_v21") -> Path:
    target = destination / name
    shutil.copytree(source / name, target)
    return target


def test_health_and_empty_registry(tmp_path: Path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/datasets").json() == {"datasets": []}
        assert client.get("/api/datasets/unknown").status_code == 404


def test_dataset_list_and_all_feature_metadata(client: TestClient, demo_root: Path):
    response = client.get("/api/datasets")
    assert response.status_code == 200
    datasets = response.json()["datasets"]
    assert {dataset["id"] for dataset in datasets} == {"demo_v21", "demo_v30"}
    assert {dataset["version"] for dataset in datasets} == {"v2.1", "v3.0"}
    for dataset in datasets:
        assert dataset["total_episodes"] == 3
        assert dataset["total_frames"] == 540
        assert dataset["fps"] == 30
        detail = client.get(f"/api/datasets/{dataset['id']}")
        assert detail.status_code == 200
        body = detail.json()
        raw = json.loads((demo_root / dataset["id"] / "meta/info.json").read_text())
        assert {feature["key"] for feature in body["features"]} == set(raw["features"])
        kinds = {feature["kind"] for feature in body["features"]}
        assert kinds == {"video", "ee", "series", "unsupported"}
        assert len(body["episodes"]) == 3
        for feature in body["features"]:
            assert {"key", "label", "kind", "dtype", "shape", "names"} <= feature.keys()


@pytest.mark.parametrize("dataset_id", ["demo_v21", "demo_v30"])
def test_episode_and_ee_downsampling(client: TestClient, dataset_id: str):
    prefix = f"/api/datasets/{dataset_id}/episodes"
    episodes = client.get(prefix)
    assert episodes.status_code == 200
    assert [episode["episode_index"] for episode in episodes.json()["episodes"]] == [0, 1, 2]
    episode = client.get(f"{prefix}/1")
    assert episode.status_code == 200
    assert episode.json()["dataset_id"] == dataset_id
    assert episode.json()["length"] == 180
    assert episode.json()["blocks"]
    complete = client.get(f"{prefix}/1/ee").json()
    sampled_response = client.get(f"{prefix}/1/ee", params={"max_points": 17})
    assert sampled_response.status_code == 200
    sampled = sampled_response.json()
    assert complete["point_count"] == complete["total_points"] == 180
    assert sampled["total_points"] == 180
    assert sampled["point_count"] == len(sampled["timestamps"]) == len(sampled["values"])
    assert 2 <= sampled["point_count"] <= 17
    assert sampled["timestamps"][0] == 0
    assert sampled["timestamps"][0] == complete["timestamps"][0]
    assert sampled["timestamps"][-1] == complete["timestamps"][-1]
    assert sampled["values"][0] == complete["values"][0]
    assert sampled["values"][-1] == complete["values"][-1]
    assert sampled["bounds"] == complete["bounds"]
    assert len(sampled["names"]) == len(sampled["units"]) == 6
    assert sampled["units"][:3] == ["m", "m", "m"]
    episode_zero = client.get(f"{prefix}/0/ee").json()
    assert episode_zero["values"] != complete["values"]


@pytest.mark.parametrize("dataset_id", ["demo_v21", "demo_v30"])
@pytest.mark.parametrize("feature", ["exterior_image_1_left", "wrist_image_left"])
def test_video_offsets_and_byte_ranges(client: TestClient, dataset_id: str, feature: str):
    endpoint = f"/api/datasets/{dataset_id}/episodes/1/video"
    response = client.get(endpoint, params={"feature": feature})
    assert response.status_code == 200
    metadata = response.json()
    assert metadata["dataset_id"] == dataset_id
    assert metadata["feature"] == feature
    assert "path" not in metadata
    assert metadata["duration_s"] == pytest.approx(6)
    assert metadata["end_time_s"] - metadata["start_time_s"] == pytest.approx(6)
    if dataset_id == "demo_v30":
        assert metadata["start_time_s"] > 0
    else:
        assert metadata["start_time_s"] == 0
    url = metadata["url"]
    full = client.get(url)
    assert full.status_code == 200
    assert full.headers["content-type"] == "video/mp4"
    assert full.headers["accept-ranges"] == "bytes"
    size = len(full.content)
    assert size > 128
    assert full.headers["content-length"] == str(size)
    assert b"ftyp" in full.content[:32]
    requested = client.get(url, headers={"Range": "bytes=16-95"})
    assert requested.status_code == 206
    assert requested.content == full.content[16:96]
    assert requested.headers["content-range"] == f"bytes 16-95/{size}"
    assert requested.headers["content-length"] == "80"
    suffix = client.get(url, headers={"Range": "bytes=-31"})
    assert suffix.status_code == 206
    assert suffix.content == full.content[-31:]
    opened = client.get(url, headers={"Range": f"bytes={size - 31}-"})
    assert opened.status_code == 206
    assert opened.content == full.content[-31:]
    clamped = client.get(url, headers={"Range": f"bytes={size - 31}-{size + 999}"})
    assert clamped.status_code == 206
    assert clamped.content == full.content[-31:]
    head = client.head(url, headers={"Range": "bytes=16-95"})
    assert head.status_code == 206
    assert head.content == b""
    assert head.headers["content-length"] == "80"


@pytest.mark.parametrize(
    "range_header",
    ["bytes=999999999999-", "bytes=90-12", "bytes=-0", "bytes=-", "bytes=0-1,4-5", "oops"],
)
def test_invalid_ranges_are_416(client: TestClient, range_header: str):
    url = "/api/datasets/demo_v21/episodes/0/video/content"
    response = client.get(url, headers={"Range": range_header})
    assert response.status_code == 416
    assert response.headers["content-range"].startswith("bytes */")
    assert response.headers["accept-ranges"] == "bytes"


@pytest.mark.parametrize(
    ("url", "status"),
    [
        ("/api/datasets/unknown", 404),
        ("/api/datasets/demo_v21/episodes/99", 404),
        ("/api/datasets/demo_v21/episodes/-1", 422),
        ("/api/datasets/demo_v21/episodes/0/ee?feature=unknown", 404),
        ("/api/datasets/demo_v21/episodes/0/ee?max_points=1", 422),
        ("/api/datasets/demo_v21/episodes/0/ee?max_points=20001", 422),
        ("/api/datasets/demo_v21/episodes/0/ee?feature=", 422),
        ("/api/datasets/demo_v21/episodes/0/video?feature=ee_pose", 422),
        ("/api/datasets/demo_v21/episodes/0/video?feature=unknown", 404),
    ],
)
def test_errors_are_actionable_and_have_no_paths(
    client: TestClient, demo_root: Path, url: str, status: int
):
    response = client.get(url)
    assert response.status_code == status
    assert "detail" in response.json()
    assert str(demo_root) not in response.text


def test_single_dataset_root_and_environment(demo_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REPLAY_DATA_ROOT", str(demo_root / "demo_v30"))
    with TestClient(create_app()) as connection:
        datasets = connection.get("/api/datasets").json()["datasets"]
        assert [dataset["id"] for dataset in datasets] == ["demo_v30"]


def test_dataset_scan_detects_added_and_moved_datasets(demo_root: Path, tmp_path: Path):
    with TestClient(create_app(tmp_path)) as connection:
        assert connection.get("/api/datasets").json() == {"datasets": []}
        first = _copy_dataset(demo_root, tmp_path)
        _copy_dataset(demo_root, tmp_path, "demo_v30")
        response = connection.get("/api/datasets")
        assert response.status_code == 200
        assert [item["id"] for item in response.json()["datasets"]] == ["demo_v21", "demo_v30"]
        (tmp_path / "not-a-dataset").mkdir()
        first.rename(tmp_path / "renamed_dataset")
        response = connection.get("/api/datasets")
        assert [item["id"] for item in response.json()["datasets"]] == ["demo_v30", "renamed_dataset"]
        assert connection.get("/api/datasets/renamed_dataset/episodes/1").status_code == 200
        assert connection.get("/api/datasets/demo_v21").status_code == 404


def test_malformed_metadata_is_422_without_local_path(demo_root: Path, tmp_path: Path):
    root = _copy_dataset(demo_root, tmp_path)
    (root / "meta/info.json").write_text("{bad json")
    with TestClient(create_app(root)) as connection:
        response = connection.get("/api/datasets/demo_v21")
        assert response.status_code == 422
        assert str(root) not in response.text


@pytest.mark.parametrize("operation", ["missing", "malformed", "escape"])
def test_episode_parquet_errors(demo_root: Path, tmp_path: Path, operation: str):
    root = _copy_dataset(demo_root, tmp_path)
    path = sorted((root / "data").rglob("*.parquet"))[0]
    if operation == "missing":
        path.unlink()
    elif operation == "malformed":
        path.write_bytes(b"not a parquet file")
    else:
        outside = tmp_path / "outside.parquet"
        shutil.copyfile(path, outside)
        path.unlink()
        path.symlink_to(outside)
    with TestClient(create_app(root)) as connection:
        response = connection.get("/api/datasets/demo_v21/episodes/0/ee")
        assert response.status_code == (404 if operation == "missing" else 422)
        assert str(tmp_path) not in response.text


@pytest.mark.parametrize("operation", ["missing", "escape"])
def test_video_file_is_confined(demo_root: Path, tmp_path: Path, operation: str):
    from replay.scripts.resolve_video import resolve_video

    root = _copy_dataset(demo_root, tmp_path)
    path = resolve_video(root, 0, "exterior_image_1_left")["path"]
    if operation == "escape":
        outside = tmp_path / "outside.mp4"
        shutil.copyfile(path, outside)
    path.unlink()
    if operation == "escape":
        path.symlink_to(outside)
    with TestClient(create_app(root)) as connection:
        for suffix in ("video", "video/content"):
            response = connection.get(f"/api/datasets/demo_v21/episodes/0/{suffix}")
            assert response.status_code == (404 if operation == "missing" else 422)
            assert str(tmp_path) not in response.text


def test_external_dataset_symlink_is_not_registered(demo_root: Path, tmp_path: Path):
    registry = tmp_path / "registered"
    registry.mkdir()
    _copy_dataset(demo_root, registry)
    outside = _copy_dataset(demo_root, tmp_path, "demo_v30")
    (registry / "external").symlink_to(outside, target_is_directory=True)
    with TestClient(create_app(registry)) as connection:
        datasets = connection.get("/api/datasets").json()["datasets"]
        assert [dataset["id"] for dataset in datasets] == ["demo_v21"]
        assert connection.get("/api/datasets/external").status_code == 404


def test_metadata_symlink_is_rejected(demo_root: Path, tmp_path: Path):
    root = _copy_dataset(demo_root, tmp_path)
    metadata = root / "meta/info.json"
    outside = tmp_path / "outside-info.json"
    shutil.copyfile(metadata, outside)
    metadata.unlink()
    metadata.symlink_to(outside)
    with TestClient(create_app(root)) as connection:
        response = connection.get("/api/datasets/demo_v21")
        assert response.status_code == 422
        assert str(tmp_path) not in response.text
