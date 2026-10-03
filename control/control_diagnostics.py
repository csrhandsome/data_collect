"""Optional, bounded background JSONL diagnostics for the VR control loop."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from control.util.pose import quat_angle_xyzw

logger = logging.getLogger(__name__)


class ControlDiagnostics:
    """Capture control timing without waiting for disk writes or a full queue."""

    def __init__(self, config, *, clock=time.monotonic_ns):
        cfg = config.get("diagnostics", {})
        self.enabled = bool(cfg.get("enabled", False))
        self.path = None
        self.error = None
        self.dropped_records = 0
        self.timings_ms = {}
        self._config = config
        self._clock = clock
        self._root = Path(cfg.get("root", "data/diagnostics"))
        sample_hz = float(cfg.get("sample_hz", 0))
        self._sample_period_ns = round(1e9 / sample_hz) if sample_hz > 0 else 0
        self._max_pending = int(cfg.get("max_pending", 1024))
        self._queue = self._thread = self._file = None
        self._stop = threading.Event()
        self._closed = False
        self._ticks_seen = self._records_written = self._samples_written = 0
        self._previous_tick_ns = self._last_log_ns = None
        self._previous_vr = (0, 0)
        self._previous_command = None

    def __enter__(self):
        if self.enabled:
            self._root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            self.path = (self._root / f"vr_control_{stamp}_{uuid4().hex[:8]}.jsonl").resolve()
            self._file = self.path.open("x", encoding="utf-8", buffering=64 * 1024)
            self._queue = queue.Queue(maxsize=self._max_pending)
            # Select only relevant configuration; robot credentials never enter the log.
            keys = {
                "control": [
                    "mode",
                    "frequency_hz",
                    "ee_filter_coeff",
                    "ee_nullspace_stiffness",
                    "max_target_translation_m",
                    "max_target_rotation_rad",
                ],
                "vr": [
                    "source",
                    "long_press_s",
                    "position_alpha",
                    "rotation_alpha",
                    "stale_after_s",
                    "translation_gain",
                    "rotation_gain",
                    "translation_limit_m",
                    "rotation_limit_rad",
                    "max_translation_speed_m_s",
                    "max_rotation_speed_rad_s",
                    "enable_rotation",
                ],
                "camera": ["fps", "width", "height", "max_age_s"],
            }
            safe_config = {
                section: {key: self._config.get(section, {}).get(key) for key in names}
                for section, names in keys.items()
            }
            self.event("run_start", schema_version=1, config=safe_config)
            self._thread = threading.Thread(
                target=self._run, name="control-diagnostics", daemon=True
            )
            try:
                self._thread.start()
            except BaseException:
                self._file.close()
                raise
            logger.info("Control diagnostics: %s", self.path)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def begin_tick(self, step, tick_ns, overruns):
        if not self.enabled:
            return
        self._ticks_seen += 1
        self._step, self._tick_ns, self._overruns = step, tick_ns, overruns
        self._tick_interval_ms = (
            (tick_ns - self._previous_tick_ns) / 1e6 if self._previous_tick_ns is not None else None
        )
        self._previous_tick_ns = tick_ns
        self._stage_ns = self._clock()
        self.timings_ms = {}

    def mark(self, stage):
        if self.enabled:
            now = self._clock()
            self.timings_ms[stage] = (now - self._stage_ns) / 1e6
            self._stage_ns = now

    def record(self, *, raw, sample, state, mapper, position, quaternion, command_sent):
        if not self.enabled or self.error is not None or self._closed:
            return
        started_ns = self._clock()
        seq, source_ns = int(sample.source_seq), int(sample.source_ns)
        old_seq, old_ns = self._previous_vr
        fresh_update = seq > 0 and seq != old_seq
        sequence_delta = seq - old_seq if old_seq > 0 and seq >= old_seq else None
        input_hz = (
            sequence_delta * 1e9 / (source_ns - old_ns)
            if sequence_delta and source_ns > old_ns > 0
            else None
        )
        self._previous_vr = (seq, source_ns)
        previous_command = self._previous_command
        if command_sent:
            self._previous_command = (
                self._tick_ns,
                np.array(position, copy=True),
                np.array(quaternion, copy=True),
            )
        if (
            self._last_log_ns is not None
            and self._tick_ns - self._last_log_ns < self._sample_period_ns
        ):
            return
        self._last_log_ns = self._tick_ns
        command_dt = (self._tick_ns - previous_command[0]) / 1e9 if previous_command else None
        record = {
            "type": "control_tick",
            "step": self._step,
            "host_monotonic_ns": self._tick_ns,
            "tick_interval_ms": self._tick_interval_ms,
            "overruns": self._overruns,
            "vr": {
                "source_monotonic_ns": source_ns,
                "source_seq": seq,
                "age_ms": (started_ns - source_ns) / 1e6 if source_ns > 0 else None,
                "new_sample": fresh_update,
                "skipped_samples": max(0, sequence_delta - 1)
                if sequence_delta is not None
                else None,
                "observed_input_hz": input_hz,
                "raw_position": getattr(raw, "raw_position", None),
                "raw_quaternion_xyzw": getattr(raw, "raw_quaternion_xyzw", None),
                "filtered_position": sample.position.tolist(),
                "filtered_quaternion_xyzw": sample.quaternion_xyzw.tolist(),
                "raw_enabled": bool(raw.arm_enabled),
                "enabled": bool(sample.enabled),
            },
            "robot": {
                "snapshot_host_monotonic_ns": int(state.sampled_monotonic_ns),
                "robot_time_s": state.robot_time_s,
                "robot_mode": state.robot_mode,
                "control_command_success_rate": state.control_command_success_rate,
                "current_errors": state.current_errors,
                "joint_position": state.joint_positions.tolist(),
                "joint_velocity": state.joint_velocities.tolist(),
                "ee_position": state.ee_position.tolist(),
                "ee_quaternion_xyzw": state.ee_quaternion_xyzw.tolist(),
                "gripper_busy": bool(state.gripper.busy),
            },
            "mapping": mapper.diagnostic_state(),
            "command": {
                "sent": bool(command_sent),
                "position": np.asarray(position).tolist(),
                "quaternion_xyzw": np.asarray(quaternion).tolist(),
                "tracking_error_m": float(np.linalg.norm(position - state.ee_position)),
                "tracking_error_rad": quat_angle_xyzw(quaternion, state.ee_quaternion_xyzw),
                "translation_speed_m_s": float(
                    np.linalg.norm(position - previous_command[1]) / command_dt
                )
                if command_sent and command_dt and command_dt > 0
                else None,
                "rotation_speed_rad_s": quat_angle_xyzw(quaternion, previous_command[2])
                / command_dt
                if command_sent and command_dt and command_dt > 0
                else None,
            },
            "timings_ms": dict(self.timings_ms),
            "dropped_records": self.dropped_records,
        }
        finished_ns = self._clock()
        record["timings_ms"]["diagnostics_prepare"] = (finished_ns - started_ns) / 1e6
        record["work_ms"] = (finished_ns - self._tick_ns) / 1e6
        self._enqueue(record)

    def event(self, name, **fields):
        if self.enabled and not self._closed:
            self._enqueue({"type": name, "host_monotonic_ns": self._clock(), **fields})

    def _enqueue(self, record):
        if self._queue is None or self.error is not None:
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped_records += 1

    def _write_record(self, record):
        self._file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")

    def _run(self):
        flushed_at = time.monotonic()
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    record = self._queue.get(timeout=0.1)
                except queue.Empty:
                    record = None
                if record is not None:
                    self._write_record(record)
                    self._records_written += 1
                    self._samples_written += record["type"] == "control_tick"
                if time.monotonic() - flushed_at >= 1.0:
                    self._file.flush()
                    flushed_at = time.monotonic()
            self._write_record(
                {
                    "type": "diagnostics_summary",
                    "control_ticks": self._ticks_seen,
                    "records_written": self._records_written,
                    "tick_samples_written": self._samples_written,
                    "dropped_records": self.dropped_records,
                }
            )
        except Exception as exc:
            self.error = exc
            logger.warning("Control diagnostics writer failed: %s", exc)
        finally:
            try:
                self._file.close()
            except Exception as exc:
                self.error = exc
                logger.warning("Closing control diagnostics failed: %s", exc)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("Control diagnostics writer is still draining: %s", self.path)
            if self.dropped_records:
                logger.warning("Control diagnostics dropped %s records", self.dropped_records)
