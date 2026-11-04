"""Real remux, atomic publication, concurrent readers, and isolated row-group I/O."""

import asyncio
import hashlib
import io
import json
import threading
from dataclasses import replace

import av
import numpy as np
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from control.collection.audio import SyntheticMicrophone
from control.collection.dataset import open_dataset
from control.collection.deletion import EpisodeDeleter
from control.collection.devices import CameraPair, FakeVR
from control.collection.recording import EpisodeRecorder
from control.collection.vr import VRResampler
from control.direct.fake import FakeBackend
from data_analysis import compact_v3
from data_analysis.dataset_io import episode_paths, episode_rows
from replay.backend.main import create_app
from replay.backend.registry import DatasetRegistry
from replay.backend.services.stream_video import stream_video
from replay.scripts.read_audio import read_audio
from replay.scripts.read_dataset import read_dataset
from replay.scripts.read_ee import read_ee
from replay.scripts.read_series import read_series
from replay.scripts.read_sidecars import read_sync, update_sync
from replay.scripts.resolve_video import resolve_video


@pytest.fixture
def collection(tmp_path, request):
    audio = getattr(request, "param", True)
    config = {
        "dataset": {"repo_id": "local/compact", "date": "test", "root": str(tmp_path),
                    "instruction": "Move", "action_space": "ee"},
        "camera": {"image_hw": 32, "fps": 30}, "control": {"frequency_hz": 100},
    }
    dataset, root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, root, config, SyntheticMicrophone({}) if audio else None)
    base = FakeBackend().snapshot()
    try:
        for index in range(3):
            recorder.start()
            for frame in range(4):
                ns = base.sampled_monotonic_ns + (index * 4 + frame) * 40_000_000
                state = replace(base, ee_position=np.array([0.4 + index * 0.1, 0.1, 0.2]),
                                sampled_monotonic_ns=ns)
                sample = VRResampler().sample(FakeVR().latest, ns)
                recorder.trace(state, sample, state.ee_position, state.ee_quaternion_xyzw)
                image = np.full((32, 32, 3), 30 + index * 70, dtype=np.uint8)
                recorder.frame(state, CameraPair(image, image, ns, ns, ns / 1e6, ns / 1e6))
            recorder.finish(state, publish=True)
    finally:
        recorder.close()
    (root / "notes.txt").write_text("Preserve this note")
    return root, config


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("collection", [False, True], indirect=True)
def test_compaction_preserves_frames_sidecars_and_resume(collection):
    root, config = collection
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["name"] = "Recorded manipulations"
    info_path.write_text(json.dumps(info))
    before = snapshot(root)
    curves = [read_ee(root, i, "observation.ee_position") for i in range(3)]
    assert compact_v3.compact_dataset(root)
    assert read_dataset(root)["name"] == "Recorded manipulations"
    assert len(set(episode_paths(root).values())) == 1
    assert len(list((root / "meta/episodes").rglob("*.parquet"))) == 1
    assert len(list((root / "videos").rglob("*.mp4"))) == 2
    after = snapshot(root)
    for name in before:
        if name.split("/")[0] not in {"meta", "videos", "data"}:
            assert before[name] == after[name]
    for index in range(3):
        assert read_ee(root, index, "observation.ee_position") == curves[index]
        source = resolve_video(root, index, "observation.exterior_image")
        assert source["start_time_s"] == pytest.approx(index * 4 / 30)
        with av.open(str(source["path"])) as container:
            stream = container.streams.video[0]
            container.seek(int(source["start_time_s"] / stream.time_base), stream=stream)
            frame = next(frame for frame in container.decode(stream)
                         if float(frame.pts * stream.time_base) >= source["start_time_s"] - 1e-6)
            assert frame.to_ndarray(format="rgb24").mean() == pytest.approx(30 + index * 70, abs=3)
        if (root / "audio").exists():
            assert read_audio(root, index)["sample_rate"] == 16000
    assert compact_v3.compact_dataset(root) is False
    resumed, _ = open_dataset(config)
    assert resumed.meta.total_episodes == 3
    resumed.finalize()
    # Deleting an interior episode still rewrites shared media and renumbers sidecars.
    EpisodeDeleter(root).delete_episode(1)
    assert len(episode_rows(root)) == 2
    assert read_ee(root, 1, "observation.ee_position")["values"] == curves[2]["values"]


@pytest.mark.parametrize("failure", ["prepare", "validate", "fsync", "commit", "interrupt"])
def test_prepare_failure_never_changes_source(collection, monkeypatch, failure):
    root, _ = collection
    before = snapshot(root)

    def fail(*args, **kwargs):
        if failure == "interrupt":
            raise KeyboardInterrupt
        raise OSError("injected compaction failure")

    if failure in ("prepare", "interrupt"):
        prepare = compact_v3._prepare

        def fail_after_prepare(*args):
            prepare(*args)
            fail()

        monkeypatch.setattr(compact_v3, "_prepare", fail_after_prepare)
    elif failure == "validate":
        monkeypatch.setattr(compact_v3, "_validate_result", fail)
    elif failure == "fsync":
        monkeypatch.setattr(compact_v3.os, "fsync", fail)
    else:
        monkeypatch.setattr(compact_v3, "_exchange_directories", fail)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else OSError):
        compact_v3.compact_dataset(root)
    assert snapshot(root) == before
    assert not list(root.parent.glob(f".{root.name}.compact-*"))


def test_read_and_annotate_while_remuxing(collection, monkeypatch):
    root, _ = collection
    ready, proceed = threading.Event(), threading.Event()
    prepare = compact_v3._prepare
    errors = []

    def paused(*args):
        prepare(*args)
        ready.set()
        assert proceed.wait(10)

    def compact():
        try:
            compact_v3.compact_dataset(root)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(compact_v3, "_prepare", paused)
    thread = threading.Thread(target=compact)
    with TestClient(create_app(root.parent)) as client:
        thread.start()
        try:
            assert ready.wait(10)
            prefix = f"/api/datasets/{root.name}/episodes/1"
            assert client.get(prefix).status_code == 200
            assert client.post(prefix + "/annotation", json={"success": False}).status_code == 200
            video = client.get(prefix + "/video", params={"feature": "observation.exterior_image"})
            assert client.get(video.json()["url"], headers={"Range": "bytes=0-127"}).status_code == 206
        finally:
            proceed.set()
            thread.join(10)
    assert not thread.is_alive() and not errors
    assert read_sync(root, 1)["success"] is False


def test_media_stream_survives_exchange_and_stale_url_is_rejected(collection):
    root, _ = collection
    registry = DatasetRegistry(root.parent)
    source = resolve_video(root, 1, "observation.exterior_image")
    old_bytes = source["path"].read_bytes()
    response = stream_video(registry, root.name, 1, "observation.exterior_image")
    with TestClient(create_app(root.parent)) as client:
        prefix = f"/api/datasets/{root.name}/episodes/1/video"
        old = client.get(prefix, params={"feature": source["feature"]}).json()
        assert compact_v3.compact_dataset(root)

        async def consume():
            return b"".join([chunk async for chunk in response.body_iterator])

        assert asyncio.run(consume()) == old_bytes
        assert client.get(old["url"], headers={"Range": "bytes=0-127"}).status_code == 409
        new = client.get(prefix, params={"feature": source["feature"]}).json()
        assert new["start_time_s"] > 0 and new["url"] != old["url"]
        media = client.get(new["url"], headers={"Range": "bytes=0-127"})
        assert media.status_code == 206 and len(media.content) == 128
        assert client.head(new["url"]).headers["accept-ranges"] == "bytes"


def test_curve_reads_only_selected_episode_row_group(collection, monkeypatch):
    root, _ = collection
    compact_v3.compact_dataset(root)
    path = episode_paths(root)[1]
    assert pq.ParquetFile(path).num_row_groups == 3
    groups = []
    read = pq.ParquetFile.read_row_groups

    def track(self, row_groups, *args, **kwargs):
        groups.append(row_groups)
        return read(self, row_groups, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_groups", track)
    series = read_series(root, 1, "observation.ee_position")
    assert series["total_points"] == 4
    assert groups == [[1]]


def test_source_change_cancels_compaction(collection, monkeypatch):
    root, _ = collection
    prepare = compact_v3._prepare

    def change_source(*args):
        prepare(*args)
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["name"] = "Changed by another writer"
        info_path.write_text(json.dumps(info))

    monkeypatch.setattr(compact_v3, "_prepare", change_source)
    with pytest.raises(RuntimeError, match="其他进程修改"):
        compact_v3.compact_dataset(root)
    assert len(set(episode_paths(root).values())) == 3


def test_annotation_changed_after_auxiliary_snapshot_is_preserved(collection, monkeypatch):
    root, _ = collection
    validate = compact_v3._validate_result

    def label_after_snapshot(*args):
        validate(*args)
        update_sync(root, 1, {"success": True})

    monkeypatch.setattr(compact_v3, "_validate_result", label_after_snapshot)
    compact_v3.compact_dataset(root)
    assert read_sync(root, 1)["success"] is True


def test_repeated_append_to_shared_mp4_keeps_every_http_episode_isolated(collection):
    root, config = collection
    colors = [30, 100, 170]
    base = FakeBackend().snapshot()
    compact_v3.compact_dataset(root)
    update_sync(root, 1, {"success": False})
    old_curves = {index: read_ee(root, index, "observation.ee_position") for index in range(3)}
    with TestClient(create_app(root.parent)) as client:
        for index in range(3, 6):
            # Each round opens a fresh writer on a dataset whose old episodes
            # already share one MP4 and one Parquet. This is the resume/append
            # path used across separate collection sessions.
            dataset, _ = open_dataset(config)
            recorder = EpisodeRecorder(dataset, root, config, SyntheticMicrophone({}))
            color = 50 + index * 25
            colors.append(color)
            try:
                recorder.start()
                for frame in range(4):
                    ns = base.sampled_monotonic_ns + (index * 4 + frame) * 40_000_000
                    state = replace(base, sampled_monotonic_ns=ns,
                                    ee_position=np.array([0.4 + index * 0.1, 0.1, 0.2]))
                    sample = VRResampler().sample(FakeVR().latest, ns)
                    recorder.trace(state, sample, state.ee_position, state.ee_quaternion_xyzw)
                    image = np.full((32, 32, 3), color, dtype=np.uint8)
                    recorder.frame(state, CameraPair(image, image, ns, ns, ns / 1e6, ns / 1e6))
                recorder.finish(state, publish=True)
            finally:
                recorder.close()
            old_curves[index] = read_ee(root, index, "observation.ee_position")
            assert len(set(episode_paths(root).values())) == 2
            assert compact_v3.compact_dataset(root)
            assert len(set(episode_paths(root).values())) == 1
            assert len(list((root / "videos").rglob("*.mp4"))) == 2
            for episode_index in range(index + 1):
                prefix = f"/api/datasets/{root.name}/episodes/{episode_index}"
                detail = client.get(prefix).json()
                assert detail["episode_index"] == episode_index and detail["length"] == 4
                metadata = client.get(prefix + "/video", params={
                    "feature": "observation.exterior_image",
                }).json()
                assert metadata["start_time_s"] == pytest.approx(episode_index * 4 / 30)
                assert metadata["duration_s"] == pytest.approx(4 / 30)
                payload = client.get(metadata["url"])
                assert payload.status_code == 200
                ranged = client.get(metadata["url"], headers={"Range": "bytes=32-95"})
                assert ranged.status_code == 206 and ranged.content == payload.content[32:96]
                # Decode the actual HTTP payload at the frontend's clip offset.
                # Every slice must contain only that episode's four frames.
                with av.open(io.BytesIO(payload.content)) as container:
                    frames = [frame for frame in container.decode(video=0)
                              if metadata["start_time_s"] - 1e-6
                              <= float(frame.pts * frame.time_base)
                              < metadata["end_time_s"] - 1e-6]
                    assert len(frames) == 4
                    for frame in frames:
                        assert frame.to_ndarray(format="rgb24").mean() == pytest.approx(
                            colors[episode_index], abs=3,
                        )
                curve = client.get(prefix + "/ee", params={
                    "feature": "observation.ee_position",
                }).json()
                assert curve["values"] == old_curves[episode_index]["values"]
                assert curve["timestamps"] == old_curves[episode_index]["timestamps"]
                assert client.get(prefix + "/audio").json()["sample_rate"] == 16000
            assert client.get(f"/api/datasets/{root.name}/episodes/1").json()["success"] is False
