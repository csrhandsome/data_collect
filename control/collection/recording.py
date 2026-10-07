"""100 Hz state trace, 30 Hz image rows and optional synchronized episode audio."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from control.collection.dataset import open_dataset
from control.collection.schema import features
from control.recording_writer import AsyncDatasetFrames, FreshCameraPair
from control.util.lerobot_util import _discard_unsaved_episode


class JsonlSink:
    def __init__(self, path):
        self.file = Path(path).open("x", encoding="utf-8")

    def add_frame(self, frame):
        self.file.write(json.dumps(frame, separators=(",", ":")) + "\n")


class EpisodeRecorder:
    def __init__(self, dataset, root, config, microphone=None):
        self.dataset, self.root, self.config = dataset, Path(root), config
        self.microphone = microphone
        self.feature_keys = set(
            features(
                int(config["camera"].get("image_hw", 224)),
                config["dataset"]["action_space"],
                config.get("tactile", {}).get("enabled", False),
                config.get("observation"),
                config.get("action"),
            )
        )
        self.writer = AsyncDatasetFrames(dataset)
        self.active = False
        self.gate = FreshCameraPair()
        self.pending = None
        self.records = []
        self.trace_writer = self.trace_sink = None
        self._closed = False

    def start(self):
        if self._closed:
            raise RuntimeError("Recorder is closed")
        if self.active:
            raise RuntimeError("Episode is already recording")
        if self.writer is None:
            # A read/edit boundary ended the previous session. Reopen only when
            # recording starts again, using fresh metadata after any deletion.
            self.dataset, _ = open_dataset(self.config)
            self.writer = AsyncDatasetFrames(self.dataset)
        self.index = int(self.dataset.meta.total_episodes)
        self.started_ns = time.monotonic_ns()
        self.trace_count = self.unique_vr = 0
        self.last_vr_seq = None
        self.trace_path = self.root / f"episode_{self.index:06d}.actions.jsonl"
        self.trace_sink = JsonlSink(self.trace_path)
        self.trace_writer = AsyncDatasetFrames(self.trace_sink, max_pending=1000)
        self.records = []
        self.pending = None
        self.gate.reset()
        self.audio_path = None
        if self.microphone is not None:
            self.audio_path = self.root / "audio" / f"episode_{self.index:06d}.wav"
            if self.audio_path.exists():
                self.trace_writer.close()
                self.trace_sink.file.close()
                raise FileExistsError(self.audio_path)
            self.microphone.default_output_path = self.audio_path
            if not self.microphone.start_recording():
                self.trace_writer.close()
                self.trace_sink.file.close()
                raise RuntimeError("Microphone refused episode start")
        self.active = True

    def trace(self, state, vr, position, quaternion):
        if not self.active:
            return
        if vr.source_seq != self.last_vr_seq:
            self.unique_vr += 1
            self.last_vr_seq = vr.source_seq
        record = {
            "sample_index": self.trace_count,
            "host_sample_monotonic_ns": state.sampled_monotonic_ns,
            "joint_position": state.joint_positions.tolist(),
            "ee_position": state.ee_position.tolist(),
            "ee_orientation_xyzw": state.ee_quaternion_xyzw.tolist(),
            "ee_pose": state.ee_pose.vector.tolist(),
            "gripper_position": state.gripper.commanded_open_ratio,
            "target_ee_position": np.asarray(position).tolist(),
            "target_ee_orientation_xyzw": np.asarray(quaternion).tolist(),
            "vr_pose_monotonic_ns": vr.source_ns,
            "vr_pose_seq": vr.source_seq,
            "vr_position": vr.position.tolist(),
            "vr_orientation_xyzw": vr.quaternion_xyzw.tolist(),
            "vr_arm_enabled": vr.enabled,
        }
        self.trace_writer.submit(record)
        self.trace_count += 1

    def frame(self, state, pair, tactile=(None, None)):
        if not self.active or not self.gate.accept(pair.front_ns, pair.wrist_ns):
            return
        self._complete(state)
        hw = int(self.config["camera"].get("image_hw", 224))
        if pair.front.shape != (hw, hw, 3) or pair.wrist.shape != (hw, hw, 3):
            raise ValueError("Camera frame shape disagrees with configured schema")
        frame = {
            "observation.joint_position": state.joint_positions.astype(np.float32),
            "observation.ee_pose": state.ee_pose.vector,
            "observation.ee_position": state.ee_position.astype(np.float32),
            "observation.gripper_position": np.array(
                [state.gripper.commanded_open_ratio], dtype=np.float32
            ),
            "task": self.config["dataset"]["instruction"],
        }
        if "observation.exterior_image" in self.feature_keys:
            frame["observation.exterior_image"] = pair.front.copy()
        if "observation.wrist_image_left" in self.feature_keys:
            frame["observation.wrist_image_left"] = pair.wrist.copy()
        tactile_keys = self.feature_keys & {
            "observation.gripper_image_left",
            "observation.gripper_image_right",
        }
        if tactile_keys:
            from control.util.img_util import center_crop_and_resize_rgb_uint8

            left, right = tactile
            for key, image in [
                ("observation.gripper_image_left", left),
                ("observation.gripper_image_right", right),
            ]:
                if key not in tactile_keys:
                    continue
                if image is None:
                    raise RuntimeError(f"Configured tactile image is unavailable: {key}")
                frame[key] = center_crop_and_resize_rgb_uint8(
                    image, out_hw=hw, crop_scale=float(self.config["camera"].get("crop_scale", 0.9))
                )
        record = {
            "frame_index": len(self.records),
            "host_frame_monotonic_ns": state.sampled_monotonic_ns,
            "external_host_capture_monotonic_ns": pair.front_ns,
            "wrist_host_capture_monotonic_ns": pair.wrist_ns,
            "external_camera_timestamp": pair.front_sensor_timestamp,
            "wrist_camera_timestamp": pair.wrist_sensor_timestamp,
            "joint_position": state.joint_positions.tolist(),
            "ee_position": state.ee_position.tolist(),
            "ee_orientation_xyzw": state.ee_quaternion_xyzw.tolist(),
            "gripper_position": state.gripper.commanded_open_ratio,
        }
        self.records.append(record)
        self.pending = (
            {
                key: value
                for key, value in frame.items()
                if key in self.feature_keys or key == "task"
            },
            record,
        )

    def _complete(self, state, terminal=False):
        if self.pending is None:
            return
        frame, record = self.pending
        for key, value in {
            "action.joint_position": state.joint_positions.astype(np.float32),
            "action.ee_pose": state.ee_pose.vector,
            "action.gripper_position": np.array(
                [state.gripper.commanded_open_ratio], dtype=np.float32
            ),
        }.items():
            if key in self.feature_keys:
                frame[key] = value
        record.update(
            {
                "action_joint_position": state.joint_positions.tolist(),
                "action_ee_pose": state.ee_pose.vector.tolist(),
                "action_gripper_position": state.gripper.commanded_open_ratio,
                "action_sample_monotonic_ns": state.sampled_monotonic_ns,
                "terminal_action": terminal,
            }
        )
        self.writer.submit(frame)
        self.pending = None

    def finish(self, state, *, save=True, success=None):
        if not self.active:
            return
        self._complete(state, terminal=True)
        self.writer.drain()
        self.trace_writer.close()
        self.trace_sink.file.close()
        saved_audio = None
        if self.microphone is not None:
            self.microphone.default_output_path = self.audio_path if save and self.records else None
            saved_audio = self.microphone.stop_recording()
        if not save or not self.records:
            _discard_unsaved_episode(self.dataset)
            self.trace_path.unlink(missing_ok=True)
            self.active = False
            self.records = []
            return
        now = time.monotonic_ns()
        sync = {
            "episode_index": self.index,
            "task": self.config["dataset"]["instruction"],
            "label": self.config["dataset"].get("label", "none"),
            "success": success,
            "control_mode": "ee",
            "action_space": self.config["dataset"]["action_space"],
            "action_target_offset_frames": 1,
            "action_semantics": "next_camera_observation_measured_state",
            "terminal_action_semantics": "final_measured_state",
            "control_frequency": self.config["control"]["frequency_hz"],
            "camera_fps": self.config["camera"]["fps"],
            "episode_start_monotonic_ns": self.started_ns,
            "episode_stop_monotonic_ns": now,
            "video_frames": len(self.records),
            "action_records": self.trace_count,
            "action_trace": self.trace_path.name,
            "vr_unique_pose_samples": self.unique_vr,
            "frame_records": self.records,
            "vad_segments": [],
            "instruction_audio_window": {
                "start_sec": None,
                "end_sec": None,
                "source": "pending",
                "audio_valid": False,
            },
        }
        if saved_audio is not None:
            sync.update(
                {
                    "audio_path": str(saved_audio.relative_to(self.root)),
                    "audio_metadata_path": str(
                        self.microphone.last_metadata_path.relative_to(self.root)
                    ),
                    "audio_start_monotonic_ns": self.microphone.audio_start_monotonic_ns,
                    "audio_stop_monotonic_ns": self.microphone.audio_stop_monotonic_ns,
                }
            )
        path = (
            self.root / "audio" if saved_audio else self.root
        ) / f"episode_{self.index:06d}.sync.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(sync, indent=2, ensure_ascii=False))
        self.dataset.save_episode()
        temporary.replace(path)
        self.active = False
        self.records = []

    def finalize(self):
        """Seal saved episodes for reading/editing; the next start resumes lazily.

        Finish or discard the active episode first. Continuous recording only
        needs save_episode(); v3 Parquet footers are written at this boundary.
        """
        if self.active:
            raise RuntimeError("Finish or discard the active episode before finalizing")
        writer, self.writer = self.writer, None
        try:
            if writer is not None:
                writer.close()
        finally:
            self._finalize_dataset()

    def _finalize_dataset(self):
        from control.util.lerobot_metadata import normalize_episode_metadata

        self.dataset.finalize()
        # Historical v2 conversions lack modern quantile columns. Normalize
        # after sealing a session so readers can load old and new shards together.
        normalize_episode_metadata(self.root)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.writer is not None:
                self.writer.close()
        finally:
            try:
                if self.trace_writer is not None:
                    self.trace_writer.close()
            finally:
                if self.trace_sink is not None:
                    self.trace_sink.file.close()
                if self.active and self.microphone is not None:
                    self.microphone.default_output_path = None
                    self.microphone.stop_recording()
                try:
                    if self.writer is not None:
                        _discard_unsaved_episode(self.dataset)
                finally:
                    self._finalize_dataset()
                    self.writer = None
                    self.active = False
