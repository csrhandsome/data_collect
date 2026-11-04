"""Newly saved episodes are playable while the collection session continues."""

import copy
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from control.collection.audio import SyntheticMicrophone
from control.collection.dataset import open_dataset
from control.collection.devices import CameraPair, FakeVR
from control.collection.pipeline import run_collection
from control.collection.recording import EpisodeRecorder
from control.config import load_config
from control.direct.fake import FakeBackend
from replay.backend.main import create_app
from replay.scripts.read_dataset import read_dataset
from replay.scripts.resolve_video import resolve_video


@pytest.mark.parametrize("audio", [False, True])
def test_publish_then_annotate_and_continue_recording(tmp_path, audio):
    config = {
        "dataset": {
            "repo_id": "local/publish",
            "date": "test",
            "root": str(tmp_path),
            "instruction": "Move",
            "action_space": "ee",
        },
        "camera": {"image_hw": 32, "fps": 30},
        "control": {"frequency_hz": 100},
    }
    dataset, root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, root, config, SyntheticMicrophone({}) if audio else None)
    backend = FakeBackend()

    def frame():
        state = backend.snapshot()
        ns = state.sampled_monotonic_ns
        image = np.full((32, 32, 3), 100, dtype=np.uint8)
        recorder.frame(state, CameraPair(image, image, ns, ns, ns / 1e6, ns / 1e6))

    try:
        with TestClient(create_app(tmp_path)) as client:
            for index in range(2):
                recorder.start()
                frame()
                frame()
                recorder.finish(backend.snapshot(), publish=True)
                assert recorder.writer is None
                assert read_dataset(root)["total_episodes"] == index + 1
                latest = client.get("/api/datasets/latest-episode").json()["episode"]
                assert latest["dataset_id"] == root.name and latest["episode_index"] == index
                prefix = f"/api/datasets/{root.name}/episodes/{index}"
                detail = client.get(prefix).json()
                assert detail["success"] is None and detail["saved_at_ns"] == latest["saved_at_ns"]
                assert (
                    client.post(f"{prefix}/annotation", json={"success": False}).status_code == 200
                )
            recorder.start()
            frame()
            # An active next episode must not prevent the previous one playing.
            response = client.get(
                f"{prefix}/video", params={"feature": "observation.exterior_image"}
            )
            assert response.status_code == 200, response.text
            assert client.get(response.json()["url"]).headers["content-type"] == "video/mp4"
            assert client.get(f"/api/datasets/{root.name}/episodes/0").json()["success"] is False
            recorder.finish(backend.snapshot(), save=False)
    finally:
        recorder.close()
    assert len(list((root / "data").rglob("*.parquet"))) == 2


@pytest.mark.parametrize("compact", [False, True])
def test_y_button_only_saves_and_publishes_unannotated_episode(tmp_path, monkeypatch, compact):
    class SaveVR(FakeVR):
        reads = 0

        @property
        def latest(self):
            sample = super().latest
            self.reads += 1
            sample.y_pressed = self.reads == 25
            # Saving returns to start and requires a fresh deadman release.
            sample.arm_enabled = self.reads != 26
            return sample

    monkeypatch.setattr("control.collection.pipeline.make_vr", lambda *_: SaveVR())
    config = copy.deepcopy(load_config())
    config["dataset"].update(root=str(tmp_path), date="button_publish", compact_on_stop=compact)
    config["camera"]["image_hw"] = 32
    config["audio"].update(enabled=False, vad_enabled=False)
    observed = []
    finish = EpisodeRecorder.finish

    def check_finish(recorder, state, **kwargs):
        index = recorder.index if recorder.active else None
        result = finish(recorder, state, **kwargs)
        if index is not None and kwargs.get("save", True):
            assert read_dataset(recorder.root)["total_episodes"] == index + 1
            payload = json.loads((recorder.root / f"episode_{index:06d}.sync.json").read_text())
            assert payload["success"] is None and payload["saved_at_ns"] > 0
            observed.append(index)
        return result

    monkeypatch.setattr(EpisodeRecorder, "finish", check_finish)
    run_collection(config, dry_run=True, max_steps=40)
    assert observed == [0, 1]
    root = tmp_path / "franka_lerobot_button_publish"
    assert len(list((root / "data").rglob("*.parquet"))) == (1 if compact else 2)
    assert len(list((root / "videos").rglob("*.mp4"))) == (2 if compact else 4)
    assert (resolve_video(root, 1, "observation.exterior_image")["start_time_s"] > 0) is compact
