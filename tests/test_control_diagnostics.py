import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from control._panda.backend import PandaBackend
from control._panda.fake import FakeBackend
from control.collection.devices import FakeVR
from control.collection.pipeline import run_collection
from control.collection.vr import EEMapper, VRResampler
from control.config import load_config, validate
from control.control_diagnostics import ControlDiagnostics
from control.vr_input import VRInputProcess


def diagnostic_config(tmp_path, **overrides):
    config = load_config()
    config["diagnostics"].update(enabled=True, root=str(tmp_path), **overrides)
    return config


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disabled_diagnostics_create_no_files_or_worker(tmp_path):
    with ControlDiagnostics({"diagnostics": {"enabled": False, "root": str(tmp_path)}}) as log:
        log.begin_tick(0, 1, 0)
        log.mark("vr_read")
        log.record(
            raw=None,
            sample=None,
            state=None,
            mapper=None,
            position=None,
            quaternion=None,
            command_sent=False,
        )
        log.event("unused")
    assert log.path is None and log._thread is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("sample_hz,expected_ticks", [(0, [0, 1, 2]), (10, [0])])
def test_logs_input_freshness_mapping_and_snapshot_without_credentials(
    tmp_path, sample_hz, expected_ticks
):
    config = diagnostic_config(tmp_path, sample_hz=sample_hz)
    config["robot"]["password"] = "must-never-be-logged"
    clock = [1_000_000_000]
    mapper, resampler = EEMapper(config["vr"]), VRResampler()
    state = FakeBackend().snapshot()
    raw = FakeVR().latest
    raw.pos_x = 0.0
    raw.raw_position = (0.0, 0.0, 0.0)
    raw.raw_quaternion_xyzw = (0.0, 0.0, 0.0, 1.0)
    with ControlDiagnostics(config, clock=lambda: clock[0]) as log:
        for tick, (seq, source_ns) in enumerate(
            [(1, 999_000_000), (1, 999_000_000), (4, 1_019_000_000)]
        ):
            clock[0] = 1_000_000_000 + tick * 10_000_000
            log.begin_tick(tick, clock[0], 0)
            clock[0] += 100_000
            log.mark("vr_read")
            raw.pose_seq, raw.pose_monotonic_ns = seq, source_ns
            if tick:
                raw.pos_x, raw.raw_position = 0.1, (0.12, 0.0, 0.0)
            sample = resampler.sample(raw, clock[0])
            position, quaternion = mapper.map(sample, state, clock[0])
            log.mark("resample_and_map")
            log.record(
                raw=raw,
                sample=sample,
                state=state,
                mapper=mapper,
                position=position,
                quaternion=quaternion,
                command_sent=True,
            )
    records = read_records(log.path)
    assert "must-never-be-logged" not in log.path.read_text()
    assert records[0]["config"]["control"]["frequency_hz"] == config["control"]["frequency_hz"]
    ticks = [row for row in records if row["type"] == "control_tick"]
    assert [row["step"] for row in ticks] == expected_ticks
    assert ticks[0]["vr"]["age_ms"] >= 1.0
    assert ticks[0]["command"]["sent"]
    assert ticks[0]["timings_ms"]["vr_read"] == pytest.approx(0.1)
    if sample_hz == 0:
        assert not ticks[1]["vr"]["new_sample"]
        assert ticks[2]["vr"]["skipped_samples"] == 2
        assert ticks[2]["vr"]["observed_input_hz"] == pytest.approx(150)
        assert ticks[2]["vr"]["raw_position"] == [0.12, 0.0, 0.0]
        assert ticks[2]["mapping"]["translation_speed_limited"]
        assert ticks[2]["mapping"]["desired_position"][0] == pytest.approx(0.5)
        assert ticks[2]["command"]["position"][0] > ticks[1]["command"]["position"][0]
    assert records[-1]["control_ticks"] == 3
    assert records[-1]["tick_samples_written"] == len(expected_ticks)
    assert records[-1]["dropped_records"] == 0


def test_slow_writer_and_full_queue_never_wait_in_the_producer(tmp_path, monkeypatch):
    log = ControlDiagnostics(diagnostic_config(tmp_path, max_pending=1))
    entered, release, produced = threading.Event(), threading.Event(), threading.Event()
    write_record = log._write_record

    def blocked_write(record):
        entered.set()
        release.wait(2.0)
        write_record(record)

    monkeypatch.setattr(log, "_write_record", blocked_write)
    with log:
        try:
            assert entered.wait(1.0)

            def produce():
                for index in range(100):
                    log.event("probe", index=index)
                produced.set()

            producer = threading.Thread(target=produce, daemon=True)
            producer.start()
            assert produced.wait(0.5), "The control producer waited for the log writer"
            producer.join(timeout=0.5)
            assert log.dropped_records > 0
        finally:
            release.set()
    assert read_records(log.path)[-1]["dropped_records"] == log.dropped_records


def test_writer_failure_does_not_raise_into_control_or_cleanup(tmp_path, monkeypatch):
    log = ControlDiagnostics(diagnostic_config(tmp_path))

    def failed_write(record):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(log, "_write_record", failed_write)
    with log:
        log._thread.join(timeout=1.0)
        assert isinstance(log.error, OSError)
        log.event("control_continues")
    assert not log._thread.is_alive()


@pytest.mark.parametrize("hz", [30, 100])
def test_collection_writes_diagnostics_without_an_active_episode(tmp_path, hz):
    config = diagnostic_config(tmp_path)
    config["control"]["frequency_hz"] = hz
    config["dataset"]["enable_logging"] = False
    config["audio"].update(enabled=False, vad_enabled=False)
    stats = run_collection(config, dry_run=True, max_steps=5)
    records = read_records(next(tmp_path.glob("vr_control_*.jsonl")))
    ticks = [row for row in records if row["type"] == "control_tick"]
    assert stats["ticks"] == len(ticks) == 5
    assert all(row["command"]["sent"] for row in ticks)
    assert all(
        set(row["timings_ms"])
        >= {
            "state_read",
            "vr_read",
            "resample_and_map",
            "scene_publish",
            "command_and_gripper",
            "recording_and_camera",
        }
        for row in ticks
    )
    assert any(row["type"] == "loop_end" for row in records)


def test_shared_memory_snapshot_includes_raw_and_filtered_pose():
    vr = VRInputProcess()
    payload = [0.0] * 25
    payload[0] = payload[7] = payload[24] = 1.0
    payload[1:4] = [0.1, 0.2, 0.3]
    payload[16:18] = [1_000_000_000, 7]
    payload[18:21] = [0.15, 0.25, 0.35]
    with vr._shm.get_lock():
        vr._shm[:] = payload
    sample = vr.latest
    assert sample.arm_enabled and sample.pose_seq == 7
    assert [sample.pos_x, sample.pos_y, sample.pos_z] == [0.1, 0.2, 0.3]
    assert sample.raw_position == (0.15, 0.25, 0.35)
    assert sample.raw_quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert vr._proc.pid is None


def test_backend_snapshot_exposes_robot_clock_and_communication_status():
    raw = SimpleNamespace(
        q=np.zeros(7),
        dq=np.zeros(7),
        O_T_EE=np.eye(4).flatten(),
        time=SimpleNamespace(to_sec=lambda: 12.5),
        robot_mode="moving",
        control_command_success_rate=0.99,
        current_errors="[]",
    )
    backend = PandaBackend({"gripper": {"type": "none"}})
    backend.panda = SimpleNamespace(get_state=lambda: raw)
    state = backend.snapshot()
    assert state.robot_time_s == 12.5 and state.robot_mode == "moving"
    assert state.control_command_success_rate == 0.99 and state.current_errors == "[]"


@pytest.mark.parametrize(
    "options",
    [{"sample_hz": -1}, {"sample_hz": float("nan")}, {"max_pending": 0}, {"max_pending": 1.5}],
)
def test_invalid_diagnostics_settings_are_rejected_before_devices_open(options):
    config = load_config()
    config["diagnostics"].update(options)
    with pytest.raises(ValueError, match="diagnostics"):
        validate(config)
