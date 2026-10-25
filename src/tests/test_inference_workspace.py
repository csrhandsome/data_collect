"""Workspace inference saves its own result and resets only after a clean stop."""

import json
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from control.base_runner import BaseRunner
from control.direct.fake import FakeBackend
from control.inference.loop import run_inference
from replay.backend.main import create_app
from replay.backend.operations import OperationManager


class FastPolicy:
    server_metadata = {
        "action_space": "ee", "action_fps": 30, "action_horizon": 50, "action_dim": 7,
    }

    def __init__(self):
        self.closed = False
        self.calls = 0

    def infer(self, observation):
        self.calls += 1
        return {"actions": np.tile(np.r_[observation["observation.ee_pose"], 1.0], (50, 1))}

    def close(self):
        self.closed = True


def config_for(root, *, audio=False):
    return {
        "dataset": {
            "repo_id": "local/inference", "date": "test", "root": str(root),
            "instruction": "Place the object", "action_space": "ee",
        },
        "robot": {"start_joint_position": [0, -0.5, 0, -2, 0, 1.5, 0.7]},
        "control": {"frequency_hz": 100},
        "camera": {"image_hw": 32, "fps": 30},
        "gripper": {"type": "none"},
        "audio": {"enabled": audio},
        "inference": {
            "host": "127.0.0.1", "port": 8001, "action_space": "ee",
            "action_fps": 30, "model_horizon": 50, "execute_steps": 10,
        },
    }


def test_vr_takeover_auto_records_actual_sources_and_resumes_policy(tmp_path, monkeypatch):
    from control.collection.vr import VRResampler

    original_sample = VRResampler.sample
    ticks = [0]

    def sample(resampler, raw, now):
        result = original_sample(resampler, raw, now)
        ticks[0] += 1
        return replace(result, enabled=20 <= ticks[0] < 40)

    monkeypatch.setattr(VRResampler, "sample", sample)

    class RLPolicy(FastPolicy):
        def infer(self, observation):
            return {
                **super().infer(observation), "source": "RL",
                "step_id": self.calls, "actor_version": "actor-1",
            }

    cfg = config_for(tmp_path)
    cfg["inference"]["enable_vr_takeover"] = True
    # Small chunks make the pre/post takeover behavior observable in a short dry run.
    cfg["inference"]["execute_steps"] = 2
    result = run_inference(cfg, dry_run=True, max_steps=70, policy=RLPolicy())
    root = tmp_path / result["episode"]["dataset_id"]
    rows = [json.loads(line) for line in (root / "episode_000000.actions.jsonl").read_text().splitlines()]
    assert len(rows) == 70
    sources = [row["control_source"] for row in rows]
    assert "RL" in sources[:19]
    assert "HUMAN" in sources[19:39]
    assert "RL" in sources[39:]
    assert all(row["intervention_flag"] == (row["control_source"] == "HUMAN") for row in rows)
    assert all("target_gripper_position" in row for row in rows)
    assert all(row["actor_version"] == "actor-1" for row in rows if row["control_source"] == "RL")
    assert all(row["policy_step_id"] is None for row in rows if row["control_source"] == "HUMAN")
    sync = json.loads((root / "episode_000000.sync.json").read_text())
    assert sync["success"] is None
    assert sync["source"] == "inference"


@pytest.mark.parametrize("human", [False, True])
@pytest.mark.parametrize("fast_gripper", [False, True])
def test_trace_matches_gripper_pause_and_resume_targets(tmp_path, monkeypatch, human, fast_gripper):
    from control.collection.vr import EEMapper, VRResampler

    entered, release = threading.Event(), threading.Event()

    class LagBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.targets = []

        def target(self, position, quaternion, nullspace):
            # Model impedance lag: an accepted target is not a measured pose.
            self.targets.append((time.monotonic_ns(), np.array(position), np.array(quaternion)))
            self.calls.append("target")

        def start(self, state):
            self.target(state.ee_position, state.ee_quaternion_xyzw, state.joint_positions)
            super().start(state)

        def gripper_command(self, *args, **kwargs):
            entered.set()
            if not fast_gripper and not release.wait(2):
                raise TimeoutError("Gripper command was not released")
            # Resuming at a changed measured pose must publish a new target.
            self.position = np.array([0.402, 0, 0.4])
            self.quaternion = np.array([0, 0, np.sin(0.01), np.cos(0.01)])
            return super().gripper_command(*args, **kwargs)

    class GripPolicy(FastPolicy):
        def infer(self, observation):
            super().infer(observation)
            pose = observation["observation.ee_pose"].copy()
            pose[0] += 0.01
            return {
                "actions": np.tile(np.r_[pose, 0.2], (50, 1)),
                "source": "RL", "step_id": self.calls, "actor_version": "actor-1",
            }

    backend = LagBackend()
    monkeypatch.setattr("control.direct.fake.FakeBackend", lambda: backend)
    submit = BaseRunner.set_gripper
    snapshot = BaseRunner.get_ee_target
    requested = False
    pause_samples = 0

    def set_gripper(arm, *args, **kwargs):
        nonlocal requested
        requested = True
        receipt = submit(arm, *args, **kwargs)
        assert entered.wait(1)
        if fast_gripper:
            # Force completion before the same tick reaches recorder.trace().
            arm.wait_gripper(timeout=1)
        return receipt

    def get_target(arm):
        nonlocal pause_samples
        target = snapshot(arm)
        if target.phase == "gripper_pause":
            pause_samples += 1
            if pause_samples == 3:
                # Complete after the loop skipped sending an EE command, so
                # the trace must read the worker's new resume target this tick.
                release.set()
                arm.wait_gripper(timeout=1)
                target = snapshot(arm)
        return target

    monkeypatch.setattr(BaseRunner, "set_gripper", set_gripper)
    monkeypatch.setattr(BaseRunner, "get_ee_target", get_target)
    if human:
        sample = VRResampler.sample

        def human_sample(resampler, raw, now):
            return replace(sample(resampler, raw, now), enabled=True, close=not requested)

        def human_target(mapper, sample, state, now):
            return state.ee_position + [0.01, 0, 0], state.ee_quaternion_xyzw.copy()

        monkeypatch.setattr(VRResampler, "sample", human_sample)
        monkeypatch.setattr(EEMapper, "map", human_target)
    cfg = config_for(tmp_path)
    cfg["gripper"]["type"] = "franka"
    cfg["inference"]["enable_vr_takeover"] = human
    try:
        result = run_inference(cfg, dry_run=True, max_steps=25, policy=GripPolicy(), record=True)
    finally:
        release.set()
    root = tmp_path / result["episode"]["dataset_id"]
    rows = [json.loads(line) for line in (root / "episode_000000.actions.jsonl").read_text().splitlines()]
    assert requested
    phases = [row["ee_target_phase"] for row in rows]
    assert "gripper_resume" in phases
    if not fast_gripper:
        assert phases.count("gripper_pause") == 2
    # Compare every recorded target to the actual backend command at that time,
    # including the command submission tick and all ticks while it is busy.
    for row in rows:
        issued, position, quaternion = next(
            entry for entry in reversed(backend.targets) if entry[0] <= row["host_sample_monotonic_ns"]
        )
        np.testing.assert_allclose(row["target_ee_position"], position)
        np.testing.assert_allclose(row["target_ee_orientation_xyzw"], quaternion)
        assert issued <= row["command_submitted_monotonic_ns"] <= row["host_sample_monotonic_ns"]
        if row["ee_target_phase"] in {"gripper_pause", "gripper_resume"}:
            assert row["control_source"] == ("HUMAN" if human else "RL")
            assert row["intervention_flag"] is human
            assert row["target_gripper_position"] == pytest.approx(0.0 if human else 0.2)
            assert row["actor_version"] == (None if human else "actor-1")
    resume_index = phases.index("gripper_resume")
    assert "command" in phases[resume_index + 1:]


def test_stop_while_connecting_does_not_open_robot(tmp_path, monkeypatch):
    stop = threading.Event()
    from control.inference.client import PolicyClient

    def refuse_connection(client):
        stop.set()
        raise ConnectionRefusedError

    def unexpected_robot():
        pytest.fail("A stopped connection attempt must not open the robot")

    monkeypatch.setattr(PolicyClient, "_connect", refuse_connection)
    monkeypatch.setattr("control.direct.fake.FakeBackend", unexpected_robot)
    assert run_inference(config_for(tmp_path), dry_run=True, stop_event=stop)["ticks"] == 0


@pytest.mark.parametrize("during_warmup", [False, True])
def test_stop_event_returns_to_start_and_closes_after_reset(tmp_path, monkeypatch, during_warmup):
    backend = FakeBackend()
    stop = threading.Event()
    monkeypatch.setattr("control.direct.fake.FakeBackend", lambda: backend)

    class StoppingPolicy(FastPolicy):
        def infer(self, observation):
            result = super().infer(observation)
            if self.calls == (2 if during_warmup else 6):
                stop.set()
            return result

    policy = StoppingPolicy()
    cfg = config_for(tmp_path)
    stats = run_inference(cfg, dry_run=True, policy=policy, stop_event=stop)
    assert policy.closed and not backend.connected
    assert backend.calls.count("move_joints") == 1
    assert backend.calls[-2:] == ["move_joints", "close"]
    np.testing.assert_allclose(backend.q, cfg["robot"]["start_joint_position"])
    if during_warmup:
        assert stats["ticks"] == 0 and "start" not in backend.calls
    else:
        assert stats["ticks"] > 0
        assert backend.calls.index("stop") < backend.calls.index("move_joints")


@pytest.mark.parametrize("reset_result", ["error", "cancelled"])
def test_return_to_start_failure_propagates(tmp_path, monkeypatch, reset_result):
    backend = FakeBackend()
    monkeypatch.setattr("control.direct.fake.FakeBackend", lambda: backend)

    def reset(arm):
        assert not backend.streaming
        if reset_result == "error":
            raise RuntimeError("reset failed")
        from types import SimpleNamespace

        return SimpleNamespace(status="cancelled")

    monkeypatch.setattr(BaseRunner, "move_to_start", reset)
    with pytest.raises(RuntimeError, match="reset failed|not completed"):
        run_inference(config_for(tmp_path), dry_run=True, max_steps=2, policy=FastPolicy())
    assert not backend.connected and backend.calls[-1] == "close"


def test_runtime_failure_stops_without_resetting_or_publishing(tmp_path, monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr("control.direct.fake.FakeBackend", lambda: backend)

    class BrokenPolicy(FastPolicy):
        def infer(self, observation):
            if self.calls >= 5:
                raise RuntimeError("policy unavailable")
            return super().infer(observation)

    with pytest.raises(RuntimeError, match="policy unavailable"):
        run_inference(config_for(tmp_path), dry_run=True, max_steps=30, record=True, policy=BrokenPolicy())
    assert "stop" in backend.calls and "move_joints" not in backend.calls
    assert not backend.connected
    assert not list(tmp_path.rglob("*.sync.json"))


@pytest.mark.parametrize("audio", [False, True])
def test_run_publishes_before_reset_and_labels_only_its_own_episode(tmp_path, monkeypatch, audio):
    backend = FakeBackend()
    monkeypatch.setattr("control.direct.fake.FakeBackend", lambda: backend)
    cfg = config_for(tmp_path, audio=audio)
    reset = BaseRunner.move_to_start
    published = []

    def check_reset(arm):
        assert not backend.streaming
        files = sorted(tmp_path.rglob("*.sync.json"))
        published.append(len(files))
        assert all(json.loads(path.read_text())["saved_at_ns"] for path in files)
        return reset(arm)

    monkeypatch.setattr(BaseRunner, "move_to_start", check_reset)
    results = [
        run_inference(cfg, dry_run=True, max_steps=12, policy=FastPolicy(), record=True)
        for _ in range(2)
    ]
    assert published == [1, 2]
    assert [item["episode"]["episode_index"] for item in results] == [0, 1]
    latest = results[-1]["episode"]
    root = tmp_path / latest["dataset_id"]
    sync_path = (root / "audio" if audio else root) / "episode_000001.sync.json"
    sync = json.loads(sync_path.read_text())
    assert sync["source"] == "inference" and sync["success"] is None
    assert sync["video_frames"] > 0 and sync["action_records"] == 12
    # Terminal action is the stopped inference state, before reset motion.
    assert sync["frame_records"][-1]["action_joint_position"] != cfg["robot"]["start_joint_position"]
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/datasets/latest-episode").json()["episode"] == latest
        prefix = f"/api/datasets/{latest['dataset_id']}/episodes"
        for success in (True, False, None):
            response = client.post(
                f"{prefix}/1/annotation",
                json={"success": success, "expected_saved_at_ns": latest["saved_at_ns"]},
            )
            assert response.status_code == 200, response.text
            assert json.loads(sync_path.read_text())["success"] is success
            assert client.get(f"{prefix}/0").json()["success"] is None


@pytest.mark.parametrize("during_warmup", [False, True])
def test_workspace_sigint_waits_for_recording_and_reset(tmp_path, during_warmup):
    """Use the real CLI/signal/supervisor boundary with an in-process fake policy."""
    import yaml

    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml.safe_dump(config_for(tmp_path / "datasets")))
    policy_script = '''
import numpy as np
import sys
import time
import control.inference.client as client

class LocalPolicy:
    server_metadata = {"action_space": "ee", "action_fps": 30, "action_horizon": 50, "action_dim": 7}
    def __init__(self, **kwargs):
        pass
    def infer(self, obs):
        time.sleep(0.05)
        return {"actions": np.tile(np.r_[obs["observation.ee_pose"], 1.0], (50, 1))}
    def close(self):
        pass

client.PolicyClient = LocalPolicy
from inference import main
main(sys.argv[1:])
'''

    class LocalPolicyManager(OperationManager):
        def start(self, kind, command, **kwargs):
            assert kind == "inference" and command[:5] == ["uv", "run", "python", "-m", "inference"]
            # Keep uv + real inference CLI; replace only the remote policy transport.
            return super().start(kind, [*command[:3], "-c", policy_script, *command[5:]], **kwargs)

    manager = LocalPolicyManager(config_path=cfg_path, simulate=True, max_steps=1000)
    with TestClient(create_app(tmp_path / "datasets", manager)) as client:
        response = client.post("/api/operations", json={"kind": "inference"})
        assert response.status_code == 202
        identifier = response.json()["id"]
        deadline = time.monotonic() + 30
        ready_marker = "Policy warmup 1/5 complete" if during_warmup else "Inference execution started"
        while not any(ready_marker in line for line in manager.snapshot().logs):
            status = manager.snapshot()
            assert status.state == "running", status.logs
            assert time.monotonic() < deadline, status.logs
            time.sleep(0.02)
        if not during_warmup:
            time.sleep(0.12)
        assert client.post("/api/operations", json={"kind": "collect"}).status_code == 409
        assert client.post(f"/api/operations/{identifier}/stop").json()["state"] == "stopping"
        # Repeated requests must not interrupt saving or return-to-start.
        assert client.post(f"/api/operations/{identifier}/stop").status_code == 200
        deadline = time.monotonic() + 30
        while manager.snapshot().state in ("running", "stopping"):
            assert time.monotonic() < deadline, manager.snapshot().logs
            time.sleep(0.02)
        status = client.get("/api/operations").json()["operation"]
        assert status["state"] == "cancelled" and status["return_code"] in (0, 130), (
            f"exit code: {status['return_code']}\n" + "\n".join(status["logs"])
        )
        assert any("Robot returned to start pose" in line for line in status["logs"])
        latest = client.get("/api/datasets/latest-episode").json()["episode"]
        assert status["result_episode"] == latest
        assert (latest is None) is during_warmup
