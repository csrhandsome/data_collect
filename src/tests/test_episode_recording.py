import json
from dataclasses import replace

import numpy as np
import pytest

from control.collection.audio import SyntheticMicrophone
from control.collection.dataset import open_dataset
from control.collection.deletion import EpisodeDeleter
from control.collection.devices import CameraPair, FakeVR
from control.collection.recording import EpisodeRecorder
from control.collection.vr import VRResampler
from control.direct.fake import FakeBackend
from data_analysis.dataset_io import episode_dataframe, episode_paths
from replay.scripts.read_dataset import read_dataset


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_ros_wrench_is_saved_in_parquet_and_traces(tmp_path, enabled):
    config = {
        "robot": {"backend": "ros2"},
        "dataset": {"repo_id": "local/wrench", "date": "capture", "root": str(tmp_path),
                    "instruction": "Contact", "action_space": "ee"},
        "camera": {"image_hw": 32, "fps": 30},
        "control": {"frequency_hz": 100},
        "observation": {
            "exterior_image": False, "wrist_image_left": False,
            "joint_position": False, "ee_pose": False, "ee_position": True,
            "gripper_position": False, "ee_wrench": enabled,
        },
        "action": {"joint_position": False, "ee_pose": False, "gripper_position": False},
    }
    dataset, root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, root, config)
    base = FakeBackend().snapshot()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    expected = []
    try:
        recorder.start()
        for index in range(3):
            wrench = np.array([index + 1, -2, 3, 0.1, -0.2, 0.3])
            expected.append(wrench)
            state = replace(base, ee_wrench=wrench, sampled_monotonic_ns=base.sampled_monotonic_ns + index)
            sample = VRResampler().sample(FakeVR().latest, state.sampled_monotonic_ns)
            recorder.trace(state, sample, state.ee_position, state.ee_quaternion_xyzw)
            ns = state.sampled_monotonic_ns
            recorder.frame(state, CameraPair(image, image, ns, ns, 0, 0))
        recorder.finish(state)
        recorder.finalize()
    finally:
        recorder.close()
    rows = episode_dataframe(episode_paths(root)[0], 0)
    sync = json.loads((root / "episode_000000.sync.json").read_text())
    trace = [json.loads(line) for line in (root / sync["action_trace"]).read_text().splitlines()]
    assert ("observation.ee_wrench" in rows) is enabled
    if enabled:
        np.testing.assert_allclose(np.stack(rows["observation.ee_wrench"]), expected, rtol=1e-6)
        assert np.stack(rows["observation.ee_wrench"]).dtype == np.float32
        np.testing.assert_allclose([row["ee_wrench"] for row in trace], expected)
        np.testing.assert_allclose([row["ee_wrench"] for row in sync["frame_records"]], expected)
    else:
        assert all("ee_wrench" not in row for row in trace + sync["frame_records"])


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("discard_between", [False, True])
def test_same_recorder_saves_consecutive_episodes(tmp_path, audio, discard_between):
    config = {
        "dataset": {
            "repo_id": "local/recording",
            "date": "consecutive",
            "root": str(tmp_path),
            "instruction": "Move the object",
            "action_space": "ee",
        },
        "camera": {"image_hw": 32, "fps": 30},
        "control": {"frequency_hz": 100},
    }
    dataset, root = open_dataset(config)
    microphone = SyntheticMicrophone({}) if audio else None
    recorder = EpisodeRecorder(dataset, root, config, microphone)
    writer = recorder.writer
    backend = FakeBackend()
    vr = FakeVR()
    resampler = VRResampler()

    def record_episode(value, save=True):
        backend.position[0] = value
        recorder.start()
        for index in range(3):
            state = backend.snapshot()
            sample = resampler.sample(vr.latest, state.sampled_monotonic_ns)
            recorder.trace(state, sample, state.ee_position, state.ee_quaternion_xyzw)
            image = np.full((32, 32, 3), index + 1, dtype=np.uint8)
            ns = state.sampled_monotonic_ns
            recorder.frame(state, CameraPair(image, image.copy(), ns, ns, ns / 1e6, ns / 1e6))
        recorder.finish(backend.snapshot(), save=save, success=save)

    try:
        record_episode(0.4)
        assert recorder.dataset.meta.total_episodes == 1
        assert recorder.dataset.meta.total_frames == 3
        if discard_between:
            record_episode(0.6, save=False)
            assert recorder.dataset.meta.total_episodes == 1
            assert not (root / "episode_000001.actions.jsonl").exists()
        record_episode(0.5)
        assert recorder.dataset is dataset and recorder.writer is writer
        recorder.finalize()
        info = read_dataset(root)
        assert info["total_episodes"] == 2
        assert info["total_frames"] == 6
        for index, path in episode_paths(root).items():
            rows = episode_dataframe(path, index, columns=["observation.ee_position", "frame_index"])
            assert rows["frame_index"].tolist() == [0, 1, 2]
            np.testing.assert_allclose(np.stack(rows["observation.ee_position"])[:, 0], 0.4 + index * 0.1)
            sync_dir = root / "audio" if audio else root
            sync = json.loads((sync_dir / f"episode_{index:06d}.sync.json").read_text())
            assert sync["video_frames"] == 3
            assert sync["action_records"] == 3
            assert sync["frame_records"][-1]["terminal_action"]
            if audio:
                assert (root / sync["audio_path"]).is_file()
        assert not list(root.rglob("*.sync.json.tmp"))
        assert len(set(episode_paths(root).values())) == 1

        # Editing a sealed session must be reflected when recording resumes.
        recorder.finalize()
        EpisodeDeleter(root).delete_episode(0)
        record_episode(0.7)
        assert recorder.dataset is not dataset
        recorder.close()
        recorder.close()
        info = read_dataset(root)
        assert info["total_episodes"] == 2 and info["total_frames"] == 6
        for index, expected in enumerate((0.5, 0.7)):
            rows = episode_dataframe(episode_paths(root)[index], index, columns=["observation.ee_position"])
            np.testing.assert_allclose(np.stack(rows["observation.ee_position"])[:, 0], expected)
        with pytest.raises(RuntimeError, match="closed"):
            recorder.start()
    finally:
        recorder.close()


def test_finalize_requires_finishing_active_episode_and_close_discards_it(tmp_path):
    config = {
        "dataset": {"repo_id": "local/recording", "date": "active", "root": str(tmp_path),
                    "instruction": "Move", "action_space": "ee"},
        "camera": {"image_hw": 32, "fps": 30},
        "control": {"frequency_hz": 100},
    }
    dataset, root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, root, config)
    try:
        recorder.start()
        with pytest.raises(RuntimeError, match="active episode"):
            recorder.finalize()
        state = FakeBackend().snapshot()
        image = np.ones((32, 32, 3), dtype=np.uint8)
        for offset in (0, 1):
            ns = state.sampled_monotonic_ns + offset
            recorder.frame(state, CameraPair(image, image, ns, ns, ns / 1e6, ns / 1e6))
    finally:
        recorder.close()
    assert dataset.meta.total_episodes == 0
    assert not list((root / "images").rglob("*.png"))


def test_trace_uses_accepted_gripper_target_and_actual_controller(tmp_path):
    config = {
        "dataset": {"repo_id": "local/recording", "date": "trace", "root": str(tmp_path),
                    "instruction": "Move", "action_space": "ee"},
        "camera": {"image_hw": 32, "fps": 30},
        "control": {"frequency_hz": 100},
    }
    dataset, root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, root, config)
    state = FakeBackend().snapshot()
    sample = VRResampler().sample(FakeVR().latest, state.sampled_monotonic_ns)
    try:
        recorder.start()
        for source in ("RL", "HUMAN", "RL"):
            recorder.trace(
                state, sample, state.ee_position, state.ee_quaternion_xyzw,
                control_source=source, target_gripper_position=0.2,
                actor_version="v2" if source == "RL" else None,
            )
        recorder.trace_writer.drain()
        recorder.trace_sink.file.flush()
        rows = [json.loads(line) for line in recorder.trace_path.read_text().splitlines()]
        assert [row["control_source"] for row in rows] == ["RL", "HUMAN", "RL"]
        assert all(row["target_gripper_position"] == 0.2 for row in rows)
        assert all(row["gripper_position"] == 1 for row in rows)
        assert [row["intervention_flag"] for row in rows] == [False, True, False]
    finally:
        recorder.close()


def test_start_backups_existing_trace_and_audio_with_copy_suffixes(tmp_path):
    config = {
        "dataset": {
            "repo_id": "local/recording",
            "date": "backup_copy",
            "root": str(tmp_path),
            "instruction": "Move",
            "action_space": "ee",
        },
        "camera": {"image_hw": 32, "fps": 30},
        "control": {"frequency_hz": 100},
    }
    dataset, root = open_dataset(config)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    trace_file = root / "episode_000000.actions.jsonl"
    audio_file = audio_dir / "episode_000000.wav"

    for expected_suffix in ("_copy", "_copy_1", "_copy_2"):
        trace_file.write_text(f"stale{expected_suffix}\n", encoding="utf-8")
        audio_file.write_bytes(f"wav{expected_suffix}".encode("utf-8"))
        recorder = EpisodeRecorder(dataset, root, config, SyntheticMicrophone({}))
        try:
            recorder.start()
            assert (root / f"episode_000000{expected_suffix}.actions.jsonl").read_text(
                encoding="utf-8"
            ) == f"stale{expected_suffix}\n"
            assert (audio_dir / f"episode_000000{expected_suffix}.wav").read_bytes() == (
                f"wav{expected_suffix}".encode("utf-8")
            )
            assert recorder.trace_path == trace_file
            assert recorder.audio_path == audio_file
        finally:
            recorder.close()
            trace_file.unlink(missing_ok=True)
            audio_file.unlink(missing_ok=True)


def test_missing_optional_sensors_record_placeholders_and_missing_metadata(tmp_path):
    from control.types import CameraPair, GripperState, RobotState

    config = {
        "dataset": {
            "repo_id": "local/recording",
            "date": "missing_sensors",
            "root": str(tmp_path),
            "instruction": "Move",
            "action_space": "ee",
        },
        "camera": {"image_hw": 16, "fps": 30},
        "control": {"frequency_hz": 100},
        "tactile": {"enabled": True},
        "observation": {"ee_wrench": True},
    }
    dataset, root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, root, config)
    state = RobotState(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        ee_position=np.array([0.4, 0.0, 0.3]),
        ee_quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        end_effector_pose=np.eye(4),
        gripper=GripperState("franka", 1.0, 0.05),
        sampled_monotonic_ns=100,
        ee_wrench=None,
    )
    pair = CameraPair(
        front=np.full((16, 16, 3), 120, dtype=np.uint8),
        wrist=np.zeros((16, 16, 3), dtype=np.uint8),
        front_ns=100,
        wrist_ns=100,
        front_sensor_timestamp=0.1,
        wrist_sensor_timestamp=0.0,
        front_valid=True,
        wrist_valid=False,
    )
    try:
        recorder.start()
        recorder.frame(state, pair, tactile=(None, None))
        recorder.finish(state, save=True)
        sync_path, = root.glob("episode_*.sync.json")
        sync = json.loads(sync_path.read_text())
        rec = sync["frame_records"][0]
        assert rec["wrist_camera_valid"] is False
        assert rec["ee_wrench"] is None
        assert rec["missing_sensors"] == [
            "ee_wrench",
            "wrist_camera",
            "observation.gripper_image_left",
            "observation.gripper_image_right",
        ]
    finally:
        recorder.close()
    dataset.finalize()
    row = dataset[0]
    assert np.isnan(row["observation.ee_wrench"].numpy()).all()
    assert np.all(row["observation.wrist_image_left"].numpy() == 0)
    assert np.all(row["observation.gripper_image_left"].numpy() == 0)


def test_recorder_saves_target_command_as_action_and_measured_state_as_observation(tmp_path):
    from control.types import EEPose

    config = {
        "dataset": {
            "repo_id": "local/recording",
            "date": "command_actions",
            "root": str(tmp_path),
            "instruction": "Push",
            "action_space": "ee",
        },
        "camera": {"image_hw": 16, "fps": 30},
        "control": {"frequency_hz": 100},
    }
    dataset, root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, root, config)
    backend = FakeBackend()
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    states = []
    targets = []
    grippers = [0.8, 0.4, 0.1]
    try:
        recorder.start()
        for index in range(2):
            backend.position[0] = 0.4 + index * 0.05
            state = backend.snapshot()
            states.append(state)
            target_pose = EEPose(
                np.array([0.45 + index * 0.05, 0.0, 0.3], dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
            targets.append(target_pose)
            ns = state.sampled_monotonic_ns + index
            recorder.frame(
                state,
                CameraPair(image, image, ns, ns, 0, 0),
                target_ee_pose=target_pose,
                target_gripper_position=grippers[index],
            )
        end_target = EEPose(
            np.array([0.55, 0.0, 0.3], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )
        targets.append(end_target)
        recorder.finish(
            backend.snapshot(),
            save=True,
            target_ee_pose=end_target,
            target_gripper_position=grippers[2],
        )
        recorder.finalize()
    finally:
        recorder.close()

    rows = episode_dataframe(episode_paths(root)[0], 0)
    sync = json.loads((root / "episode_000000.sync.json").read_text())
    assert sync["action_semantics"] == "next_camera_observation_target_command"
    assert sync["terminal_action_semantics"] == "final_target_command"
    np.testing.assert_allclose(np.stack(rows["observation.ee_pose"]), [s.ee_pose.vector for s in states], atol=1e-6)
    np.testing.assert_allclose(np.stack(rows["action.ee_pose"]), [targets[1].vector, targets[2].vector], atol=1e-6)
    np.testing.assert_allclose(np.asarray(rows["action.gripper_position"], dtype=np.float32).reshape(-1), [0.4, 0.1], atol=1e-6)
    np.testing.assert_allclose(
        [rec["action_ee_pose"] for rec in sync["frame_records"]],
        [targets[1].vector, targets[2].vector],
        atol=1e-6,
    )
    np.testing.assert_allclose([rec["action_gripper_position"] for rec in sync["frame_records"]], [0.4, 0.1], atol=1e-6)
