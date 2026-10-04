import json

import numpy as np
import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from control.collection.dataset import open_dataset
from control.collection.pipeline import run_collection
from control.config import load_config
from control.reactive_desk_client import ReactiveDeskVlaClient, ScenePublisher


@pytest.fixture
def local_collection_config(tmp_path, monkeypatch):
    from lerobot.datasets import dataset_metadata

    def forbid_hub(*args, **kwargs):
        pytest.fail("Local recording must not fall back to Hugging Face")

    monkeypatch.setattr(dataset_metadata, "get_safe_version", forbid_hub)
    monkeypatch.setattr(dataset_metadata.LeRobotDatasetMetadata, "_pull_from_repo", forbid_hub)
    config = load_config()
    config["dataset"].update(root=str(tmp_path), date="startup")
    return config


def test_resume_before_first_saved_episode(local_collection_config, tmp_path):
    dataset, root = open_dataset(local_collection_config)
    assert root == tmp_path / "franka_lerobot_startup"
    assert dataset.repo_id == "openpi/franka_lerobot_startup"
    dataset.finalize()
    info_before = (root / "meta/info.json").read_bytes()

    for _ in range(2):
        resumed, resumed_root = open_dataset(local_collection_config)
        try:
            assert resumed_root == root
            assert resumed.meta.total_episodes == 0
            assert not resumed.has_pending_frames()
            assert (root / "meta/info.json").read_bytes() == info_before
            assert not (root / "meta/tasks.parquet").exists()
            assert not (root / "meta/episodes").exists()
        finally:
            resumed.finalize()


@pytest.mark.parametrize("count", ["total_episodes", "total_frames", "total_tasks"])
def test_missing_recorded_metadata_is_not_replaced(local_collection_config, count):
    dataset, root = open_dataset(local_collection_config)
    dataset.finalize()
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info[count] = 1
    info_path.write_text(json.dumps(info))

    with pytest.raises(RuntimeError, match="missing local metadata"):
        open_dataset(local_collection_config)
    assert not (root / "meta/tasks.parquet").exists()
    assert not (root / "meta/episodes").exists()


def test_resume_preserves_saved_episodes(local_collection_config):
    local_collection_config["audio"].update(enabled=False, vad_enabled=False)
    for _ in range(2):
        run_collection(local_collection_config, dry_run=True, max_steps=30)
    dataset, root = open_dataset(local_collection_config)
    try:
        assert dataset.meta.total_episodes == 2
        for index in range(2):
            assert (root / dataset.meta.get_data_file_path(index)).is_file()
        assert dataset.meta.total_frames > 0
    finally:
        dataset.finalize()


def test_scene_handshake_failure_does_not_interrupt_collection(monkeypatch):
    def reject_handshake(*args, **kwargs):
        raise InvalidStatus(Response(404, "Not Found", Headers()))

    monkeypatch.setattr("websockets.sync.client.connect", reject_handshake)
    client = ReactiveDeskVlaClient()
    assert not client.send_predictions(np.zeros(3), np.ones(1))
    assert client._ws is None

    scene = ScenePublisher({"reactive_desk": {"reactive_desk_enabled": True}})
    try:
        scene.publish(np.zeros(3), 1_000_000_000, "test")
        assert scene.future.result(timeout=2) is False
        scene.publish(np.zeros(3), 2_000_000_000, "test")
        assert scene.future.result(timeout=2) is False
    finally:
        scene.close()


@pytest.mark.parametrize(
    "moving,audio_enabled,interrupt",
    [(False, True, False), (False, True, True), (True, False, False), (True, True, False)],
)
def test_vad_requires_saved_audio(
    local_collection_config, monkeypatch, moving, audio_enabled, interrupt
):
    from control._panda.fake import FakeBackend
    from control.collection import pipeline
    from control.collection.audio import SyntheticMicrophone
    from control.collection.devices import FakeCameras, FakeVR
    from data_analysis import preprocess_vad

    arm_factory = pipeline.RoboticArmControler
    scene_factory = pipeline.ScenePublisher
    monkeypatch.setattr(
        pipeline,
        "RoboticArmControler",
        lambda config, backend: arm_factory(config=config, backend=FakeBackend()),
    )
    monkeypatch.setattr(pipeline, "make_cameras", lambda cfg, dry_run: FakeCameras(cfg))

    class TestVR(FakeVR):
        @property
        def latest(self):
            if interrupt:
                raise KeyboardInterrupt
            sample = super().latest
            sample.arm_enabled = moving
            return sample

    monkeypatch.setattr(pipeline, "make_vr", lambda cfg, dry_run: TestVR())
    monkeypatch.setattr(
        pipeline,
        "make_microphone",
        lambda cfg, dry_run: SyntheticMicrophone(cfg["audio"]) if audio_enabled else None,
    )
    monkeypatch.setattr(
        pipeline, "ScenePublisher", lambda cfg, dry_run: scene_factory(cfg, dry_run=True)
    )
    calls = []
    monkeypatch.setattr(preprocess_vad, "compute_vad_for_dataset", lambda root: calls.append(root))
    local_collection_config["audio"].update(enabled=audio_enabled, vad_enabled=True)

    result = run_collection(local_collection_config, dry_run=False, max_steps=30)

    assert result["ticks"] == (0 if interrupt else 30)
    dataset, root = open_dataset(local_collection_config)
    try:
        assert dataset.meta.total_episodes == int(moving)
        assert calls == ([root] if moving and audio_enabled else [])
        if moving and audio_enabled:
            assert (root / "audio/episode_000000.wav").is_file()
        else:
            assert not (root / "audio").exists()
    finally:
        dataset.finalize()
