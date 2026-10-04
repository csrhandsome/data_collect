"""Exercise real subprocess supervision and deletion only on temporary datasets."""

import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from control._panda.fake import FakeBackend
from control.collection import EpisodeDeleter
from control.robotic_arm_controller import RoboticArmControler
from replay.backend.errors import ReplayError
from replay.backend.main import create_app
from replay.backend.operations import OperationManager
from replay.scripts.read_dataset import read_dataset
from replay.scripts.replay_robot import load_trajectory, replay_trajectory


def test_collect_console_reports_success(monkeypatch):
    import vr_collect

    monkeypatch.setattr(vr_collect, "main", lambda: {"ticks": 10, "overruns": 0})
    assert vr_collect.cli() == 0


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory):
    from replay.scripts.generate_demo import generate_demo

    root = tmp_path_factory.mktemp("operations-demo")
    generate_demo(root)
    return root


def wait_finished(manager):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = manager.snapshot()
        if status and status.state not in ("running", "stopping"):
            return status
        time.sleep(0.02)
    pytest.fail("Task did not finish")


def copy_dataset(demo_root, tmp_path):
    import shutil

    target = tmp_path / "demo_v21"
    shutil.copytree(demo_root / "demo_v21", target)
    return target


def test_task_logs_failure_and_bounded_history():
    manager = OperationManager()
    try:
        manager.start(
            "collect", [sys.executable, "-c", "for i in range(100): print(i)\nraise SystemExit(7)"]
        )
        finished = wait_finished(manager)
        assert finished.state == "failed"
        assert finished.return_code == 7
        assert finished.finished_at
        assert len(finished.logs) == 80 and finished.logs[-1] == "99"
    finally:
        manager.close()


def test_task_conflict_stop_and_cleanup():
    manager = OperationManager()
    try:
        operation = manager.start(
            "replay",
            [
                sys.executable,
                "-u",
                "-c",
                "import signal,time,sys\nsignal.signal(signal.SIGINT, lambda *_: sys.exit(0))\nprint('READY')\ntime.sleep(20)",
            ],
        )
        deadline = time.monotonic() + 5
        while "READY" not in manager.snapshot().logs and time.monotonic() < deadline:
            time.sleep(0.02)
        assert "READY" in manager.snapshot().logs
        with pytest.raises(ReplayError, match="已有任务"):
            manager.start("collect", [sys.executable, "-c", "pass"])
        assert manager.stop(operation.id).state == "stopping"
        assert wait_finished(manager).state == "cancelled"
        next_operation = manager.start("collect", [sys.executable, "-c", "print('ok')"])
        assert next_operation.id != operation.id
        assert wait_finished(manager).state == "succeeded"
    finally:
        manager.close()


class RecordingManager(OperationManager):
    def start(self, kind, command, **kwargs):
        self.last_command = command
        # Route tests cannot open any real hardware.
        return super().start(kind, [sys.executable, "-c", "pass"], **kwargs)


def test_api_uses_fixed_commands_and_latest_episode(demo_root):
    manager = RecordingManager()
    with TestClient(create_app(demo_root, manager)) as client:
        assert client.get("/api/operations").json() == {"operation": None}
        started = client.post("/api/operations", json={"kind": "collect"})
        assert started.status_code == 202
        assert manager.last_command == ["uv", "run", "vr_collect"]
        wait_finished(manager)
        response = client.post(
            "/api/operations", json={"kind": "replay", "dataset_id": "demo_v21", "episode_index": 2}
        )
        assert response.status_code == 202
        assert response.json()["episode_index"] == 2
        assert "replay.scripts.replay_robot" in manager.last_command
        assert manager.last_command[-2:] == ["--episode-index", "2"]
        wait_finished(manager)
        stale = client.post(
            "/api/operations", json={"kind": "replay", "dataset_id": "demo_v21", "episode_index": 0}
        )
        assert stale.status_code == 409
        assert (
            client.post(
                "/api/operations", json={"kind": "collect", "command": "danger"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/operations",
                json={"kind": "delete", "dataset_id": "demo_v30", "episode_index": 1},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/operations",
                json={"kind": "delete", "dataset_id": "demo_v21", "episode_index": 100},
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/operations", json={"kind": "replay", "dataset_id": "../demo_v21"}
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/operations",
                json={"kind": "collect"},
                headers={"Origin": "https://other.example"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/operations", json={"kind": "collect"}, headers={"Origin": "http://testserver"}
            ).status_code
            == 202
        )


@pytest.mark.parametrize("custom_video_path", [False, True])
def test_delete_middle_episode_and_new_sidecars(demo_root, tmp_path, custom_video_path):
    root = copy_dataset(demo_root, tmp_path)
    if custom_video_path:
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["video_path"] = "clips/{video_key}/clip_{episode_index:06d}.mp4"
        info["total_videos"] = 6
        for key, feature in info["features"].items():
            if feature["dtype"] != "video":
                continue
            for index in range(3):
                destination = root / f"clips/{key}/clip_{index:06d}.mp4"
                destination.parent.mkdir(parents=True, exist_ok=True)
                (root / f"videos/chunk-000/{key}/episode_{index:06d}.mp4").rename(destination)
        info_path.write_text(json.dumps(info))
    (root / "audio").mkdir()
    for index in range(3):
        stem = f"episode_{index:06d}"
        (root / f"{stem}.actions.jsonl").write_text(json.dumps({"marker": index}) + "\n")
        (root / f"{stem}.sync.json").write_text(
            json.dumps({"episode_index": index, "action_trace": f"{stem}.actions.jsonl"})
        )
        (root / "audio" / f"{stem}.wav").write_bytes(bytes([index]))
        (root / "audio" / f"{stem}.audio.json").write_text(
            json.dumps({"audio_path": str(root / "audio" / f"{stem}.wav")})
        )
    original = pq.read_table(root / "data/chunk-000/episode_000002.parquet")
    assert EpisodeDeleter(root).preview_episode(1).dry_run
    assert read_dataset(root)["total_episodes"] == 3
    assert not EpisodeDeleter(root).delete_episode(1).dry_run
    detail = read_dataset(root)
    assert detail["total_episodes"] == 2 and detail["total_frames"] == 360
    if custom_video_path:
        info = json.loads((root / "meta/info.json").read_text())
        assert info["total_videos"] == 4
        assert (root / "clips/wrist_image_left/clip_000001.mp4").is_file()
        assert not (root / "clips/wrist_image_left/clip_000002.mp4").exists()
    shifted = pq.read_table(root / "data/chunk-000/episode_000001.parquet")
    assert shifted["episode_index"].to_pylist() == [1] * 180
    assert shifted["index"].to_pylist() == list(range(180, 360))
    assert shifted["ee_pose"].equals(original["ee_pose"])
    assert json.loads((root / "episode_000001.actions.jsonl").read_text())["marker"] == 2
    sync = json.loads((root / "episode_000001.sync.json").read_text())
    assert sync["episode_index"] == 1 and sync["action_trace"] == "episode_000001.actions.jsonl"
    assert "audio_path" not in sync
    assert not (root / "episode_000002.actions.jsonl").exists()
    assert not (root / "episode_000002.sync.json").exists()
    assert (root / "audio/episode_000001.wav").read_bytes() == b"\x02"
    stats = json.loads((root / "meta/stats.json").read_text())
    assert stats["index"]["max"] == [359]
    assert stats["index"]["count"] == [360]
    assert not EpisodeDeleter(root).delete_episode(1).dry_run
    assert not EpisodeDeleter(root).delete_episode(0).dry_run
    assert read_dataset(root)["total_episodes"] == 0


def test_delete_without_audio_and_reject_bad_inputs_before_writes(demo_root, tmp_path):
    root = copy_dataset(demo_root, tmp_path)
    (root / "episode_000001.actions.jsonl").write_text("{}\n")
    assert not EpisodeDeleter(root).delete_episode(1).dry_run
    assert not (root / "episode_000001.actions.jsonl").exists()
    broken = root / "data/chunk-000/episode_000001.parquet"
    broken.write_bytes(b"broken")
    original_info = (root / "meta/info.json").read_bytes()
    with pytest.raises(Exception):
        EpisodeDeleter(root).delete_episode(0)
    assert (root / "data/chunk-000/episode_000000.parquet").exists()
    assert (root / "meta/info.json").read_bytes() == original_info


def test_delete_rejects_symlink_and_path_escape(demo_root, tmp_path):
    root = copy_dataset(demo_root, tmp_path)
    (root / "unsafe").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="符号链接"):
        EpisodeDeleter(root).delete_episode(1)
    (root / "unsafe").unlink()
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["data_path"] = "../episode_{episode_index:06d}.parquet"
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="unsafe"):
        EpisodeDeleter(root).delete_episode(1)
    assert (root / "data/chunk-000/episode_000001.parquet").exists()


def test_replay_loads_both_versions_and_preserves_pose(demo_root):
    for dataset in ("demo_v21", "demo_v30"):
        index, points = load_trajectory(demo_root / dataset)
        assert index == 2 and len(points) == 180
        assert points[0]["time_s"] == 0
        assert points[-1]["time_s"] > 5
        assert len(points[0]["quaternion_xyzw"]) == 4


def test_replay_prefers_trace_and_executes_gripper_with_fake_arm(demo_root, tmp_path):
    root = copy_dataset(demo_root, tmp_path)
    rows = [
        {
            "host_sample_monotonic_ns": 1_000_000_000 + i * 10_000_000,
            "ee_position": [0.4 + i * 0.001, 0, 0.4],
            "ee_orientation_xyzw": [0, 0, 0, 1],
            "gripper_position": float(i == 0),
        }
        for i in range(3)
    ]
    (root / "episode_000002.actions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (root / "episode_000002.sync.json").write_text(
        json.dumps({"episode_index": 2, "action_trace": "episode_000002.actions.jsonl"})
    )
    _, points = load_trajectory(root)
    assert len(points) == 3 and points[-1]["time_s"] == 0.02
    backend = FakeBackend()
    with RoboticArmControler(config={"gripper": {"type": "franka"}}, backend=backend) as arm:
        replay_trajectory(arm, points)
        np.testing.assert_allclose(arm.get_state().ee_position, rows[-1]["ee_position"])
    assert backend.calls.count("gripper") == 2
    assert "stop" in backend.calls and backend.calls[-1] == "close"
    rows[1]["host_sample_monotonic_ns"] = rows[0]["host_sample_monotonic_ns"]
    (root / "episode_000002.actions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="严格递增"):
        load_trajectory(root)


def test_delete_api_runs_public_deleter_on_temporary_copy(demo_root, tmp_path):
    root = copy_dataset(demo_root, tmp_path)

    class DeleteManager(OperationManager):
        def start(self, kind, command, **kwargs):
            assert kind == "delete"
            assert command[:3] == ["uv", "run", "episode-delete"]
            return super().start(
                kind,
                [
                    sys.executable,
                    "-c",
                    "from control.collection.deletion import cli; raise SystemExit(cli())",
                    *command[3:],
                ],
                **kwargs,
            )

    manager = DeleteManager()
    with TestClient(create_app(tmp_path, manager)) as client:
        response = client.post(
            "/api/operations", json={"kind": "delete", "dataset_id": root.name, "episode_index": 1}
        )
        assert response.status_code == 202
        assert wait_finished(manager).state == "succeeded"
        detail = client.get(f"/api/datasets/{root.name}").json()
        assert detail["total_episodes"] == 2
        assert client.get(f"/api/datasets/{root.name}/episodes/1").status_code == 200
