"""Human annotations survive reload, audio processing and episode discovery."""

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from replay.backend.main import create_app
from replay.backend.operations import Operation, OperationManager
from replay.scripts.read_sidecars import update_sync


@pytest.mark.parametrize("name", ["demo_v21", "demo_v30"])
@pytest.mark.parametrize("folder", ["", "audio"])
def test_annotation_preserves_episode_and_sidecar(copy_dataset, name, folder):
    root = copy_dataset(name)
    directory = root / folder
    directory.mkdir(exist_ok=True)
    path = directory / "episode_000001.sync.json"
    original = {
        "episode_index": 1,
        "success": None,
        "label": "detour",
        "saved_at_ns": 1_800_000_000_000_000_123,
        "vad_segments": [{"start_sec": 0.1, "end_sec": 0.3}],
        "custom": {"keep": True},
    }
    path.write_text(json.dumps(original))
    files = [*root.rglob("*.parquet"), root / "meta/info.json"]
    digests = {file: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    prefix = f"/api/datasets/{name}/episodes/1"
    with TestClient(create_app(root)) as client:
        assert client.get(prefix).json()["success"] is None
        for value in (False, True, None):
            response = client.post(
                f"{prefix}/annotation",
                json={
                    "success": value,
                    "expected_saved_at_ns": str(original["saved_at_ns"]),
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["success"] is value
            assert client.get(prefix).json()["success"] is value
            assert json.loads(path.read_text()) == {**original, "success": value}
        client.post(f"{prefix}/annotation", json={"success": False})
        latest = client.get("/api/datasets/latest-episode").json()["episode"]
        assert latest == {
            "dataset_id": name,
            "episode_index": 1,
            "saved_at_ns": str(original["saved_at_ns"]),
        }
        # Audio jobs merge only their fields; they retain the newer human label.
        update_sync(root, 1, {"vad_segments": []})
        update_sync(root, 1, {"instruction_audio_window": {"audio_valid": False}})
        assert client.get(prefix).json()["success"] is False
        assert client.get("/api/datasets/latest-episode").json()["episode"] == latest
    assert digests == {file: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    assert not list(directory.glob("*.sync.tmp"))
    with TestClient(create_app(root)) as client:
        assert client.get(prefix).json()["success"] is False


def test_legacy_episode_without_sidecar_can_be_annotated(copy_dataset):
    root = copy_dataset()
    with TestClient(create_app(root)) as client:
        prefix = "/api/datasets/demo_v21/episodes/0"
        assert client.get(prefix).json()["success"] is None
        assert client.get("/api/datasets/latest-episode").json() == {"episode": None}
        assert client.post(f"{prefix}/annotation", json={"success": False}).status_code == 200
        assert json.loads((root / "episode_000000.sync.json").read_text()) == {
            "episode_index": 0,
            "success": False,
        }
        assert client.get("/api/datasets/latest-episode").json() == {"episode": None}


@pytest.mark.parametrize(
    "payload",
    [{}, {"success": 0}, {"success": "false"}, {"success": True, "path": "/tmp/arbitrary"}],
)
def test_annotation_requires_explicit_boolean_or_null(copy_dataset, payload):
    root = copy_dataset()
    with TestClient(create_app(root)) as client:
        response = client.post("/api/datasets/demo_v21/episodes/0/annotation", json=payload)
        assert response.status_code == 422
    assert not list(root.rglob("*.sync.json"))


def test_unknown_episode_origin_and_stale_episode_are_rejected(copy_dataset):
    root = copy_dataset()
    update_sync(root, 0, {"saved_at_ns": 123, "success": None})
    with TestClient(create_app(root)) as client:
        prefix = "/api/datasets/demo_v21/episodes"
        assert client.post(f"{prefix}/99/annotation", json={"success": True}).status_code == 404
        assert (
            client.post(
                f"{prefix}/0/annotation",
                json={"success": True},
                headers={"Origin": "http://other-site"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                f"{prefix}/0/annotation",
                json={
                    "success": True,
                    "expected_saved_at_ns": "122",
                },
            ).status_code
            == 409
        )
        assert client.get(f"{prefix}/0").json()["success"] is None


def test_sidecar_symlink_cannot_be_written(copy_dataset, tmp_path: Path):
    root = copy_dataset()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"episode_index": 0, "success": None}))
    (root / "episode_000000.sync.json").symlink_to(outside)
    with TestClient(create_app(root)) as client:
        response = client.post(
            "/api/datasets/demo_v21/episodes/0/annotation", json={"success": True}
        )
        assert response.status_code == 422
        assert str(tmp_path) not in response.text
    assert json.loads(outside.read_text())["success"] is None


def test_annotation_waits_for_deletion(copy_dataset):
    root = copy_dataset()
    manager = OperationManager()
    manager._operation = Operation(id="deleting", kind="delete", state="running", started_at="now")
    with TestClient(create_app(root, manager)) as client:
        response = client.post(
            "/api/datasets/demo_v21/episodes/0/annotation", json={"success": False}
        )
        assert response.status_code == 409
    assert not list(root.rglob("*.sync.json"))


def test_latest_episode_uses_publication_time_across_datasets(copy_dataset, tmp_path: Path):
    first = copy_dataset()
    second = copy_dataset("demo_v30")
    update_sync(first, 2, {"saved_at_ns": 100})
    update_sync(second, 0, {"saved_at_ns": 200})
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/datasets/latest-episode")
        assert response.status_code == 200
        assert response.json()["episode"] == {
            "dataset_id": "demo_v30",
            "episode_index": 0,
            "saved_at_ns": "200",
        }
        update_sync(first, 2, {"success": True})
        assert client.get("/api/datasets/latest-episode").json() == response.json()
