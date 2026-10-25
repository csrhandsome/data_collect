"""Recursive YAML loading and validation before any device is opened."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import yaml

from control.collection.schema import features
from control.util.timing import positive_rate

DEFAULT_CONFIG = Path(
    os.environ.get(
        "DATA_COLLECT_CONFIG", Path(__file__).resolve().parents[2] / "config/train/panda.yaml"
    )
)
INFERENCE_CONFIG = Path(os.environ.get(
    "DATA_COLLECT_INFERENCE_CONFIG",
    Path(__file__).resolve().parents[2] / "config/inference/panda.yaml",
))


def load_config(path=DEFAULT_CONFIG, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError("Cyclic YAML inheritance")
    seen.add(path)
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a YAML mapping")
    parent = payload.pop("extends", None)
    result = load_config(path.parent / parent, seen) if parent else {}
    merge(result, payload)
    # Validate the merged child: inherited field choices may depend on its
    # tactile/action-space settings rather than the defaults of a partial parent.
    if _seen is None:
        # Resolve backend defaults only after inheritance, so changing backend in
        # a child config also changes FCI ownership, reset and telemetry defaults.
        robot = result.setdefault("robot", {})
        backend = robot.setdefault("backend", "panda")
        robot.setdefault("activate_fci", backend == "panda")
        robot.setdefault("move_to_start", backend == "panda")
        result.setdefault("ros2", {}).setdefault("enabled", backend == "ros2")
        for section in ("camera", "audio"):
            result.setdefault(section, {}).setdefault(
                "backend", "ros2" if backend == "ros2" else "direct"
            )
        validate(result)
    return result


def merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def validate(config):
    robot = config.get("robot", {})
    backend = robot.get("backend", "panda")
    if backend not in ("panda", "ros2"):
        raise ValueError("robot.backend must be panda or ros2")
    for section in ("camera", "audio"):
        cfg = config.get(section, {})
        default = "ros2" if backend == "ros2" else "direct"
        if cfg.get("backend", default) not in ("direct", "ros2"):
            raise ValueError(f"{section}.backend must be direct or ros2")
    if config.get("camera", {}).get("backend") == "ros2":
        cfg = config["camera"]
        for key, default in (("max_age_s", 0.2), ("startup_timeout_s", 10)):
            positive_rate(cfg.get(key, default), f"camera.{key}")
        ros = cfg.get("ros2", {})
        positive_rate(ros.get("max_pair_skew_s", 0.1), "camera.ros2.max_pair_skew_s")
        for key in ("external_image_topic", "wrist_image_topic"):
            if key in ros and (not isinstance(ros[key], str) or not ros[key].strip()):
                raise ValueError(f"camera.ros2.{key} must be a nonempty string")
    if config.get("audio", {}).get("backend") == "ros2":
        ros = config["audio"].get("ros2", {})
        for key, default in (("max_age_s", 0.5), ("startup_timeout_s", 10)):
            positive_rate(ros.get(key, default), f"audio.ros2.{key}")
        if "topic" in ros and (not isinstance(ros["topic"], str) or not ros["topic"].strip()):
            raise ValueError("audio.ros2.topic must be a nonempty string")
        channels = config["audio"].get("channels", 1)
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError("audio.channels must be a positive integer")
    if backend == "ros2":
        if robot.get("move_to_start", True):
            raise ValueError("ROS equilibrium control requires robot.move_to_start=false")
        if robot.get("activate_fci", False):
            raise ValueError("ROS backend leaves FCI ownership to franka_ros2; disable activate_fci")
        if config.get("inference", {}).get("move_to_start", False):
            raise ValueError("ROS backend requires inference.move_to_start=false")
        if config.get("inference", {}).get("action_space", "ee") != "ee":
            raise ValueError("ROS equilibrium control requires inference.action_space=ee")
        cfg = config.get("ros2", {})
        for key, default in (("state_timeout_s", 0.25), ("connect_timeout_s", 10),
                             ("gripper_timeout_s", 10)):
            positive_rate(cfg.get(key, default), f"ros2.{key}")
        for key in ("base_frame", "arm_id"):
            if not isinstance(cfg.get(key), str) or not cfg[key].strip():
                raise ValueError(f"ros2.{key} must be a nonempty string")
    control = config.get("control", {})
    if control.get("mode", "ee") != "ee":
        raise ValueError("control.mode must be ee; action fields select saved joint/ee labels")
    if config.get("dataset", {}).get("action_space", "joint") not in ("joint", "ee"):
        raise ValueError("dataset.action_space must be joint or ee")
    if "image_storage" in config.get("dataset", {}):
        raise ValueError("dataset.image_storage was removed; camera features always use MP4")
    if "fields" in config.get("dataset", {}):
        raise ValueError("dataset.fields was replaced by observation/action mappings")
    features(
        int(config.get("camera", {}).get("image_hw", 224)),
        config.get("dataset", {}).get("action_space", "joint"),
        config.get("tactile", {}).get("enabled", False),
        config.get("observation"),
        config.get("action"),
    )
    if "video_encoder" in config.get("dataset", {}):
        from lerobot.configs import RGBEncoderConfig

        RGBEncoderConfig(**config["dataset"]["video_encoder"])
    if not float(config.get("camera", {}).get("fps", 30)).is_integer():
        raise ValueError("Video recording requires an integer camera.fps with this LeRobot version")
    if config.get("gripper", {}).get("type", "franka") not in ("franka", "dh5", "none"):
        raise ValueError("gripper.type must be franka, dh5 or none")
    for section, key, default in [
        ("control", "frequency_hz", 100),
        ("camera", "fps", 30),
        ("inference", "action_fps", 30),
        ("audio", "sample_rate", 16000),
    ]:
        positive_rate(config.get(section, {}).get(key, default), f"{section}.{key}")
    inference = config.get("inference")
    if inference is not None:
        model_horizon = inference.get("model_horizon")
        execute_steps = inference.get("execute_steps")
        if (
            isinstance(model_horizon, bool)
            or not isinstance(model_horizon, int)
            or model_horizon <= 0
        ):
            raise ValueError("inference.model_horizon must be a positive integer")
        if (
            isinstance(execute_steps, bool)
            or not isinstance(execute_steps, int)
            or not 1 <= execute_steps <= model_horizon
        ):
            raise ValueError("inference.execute_steps must be an integer in 1..model_horizon")
        if "action_horizon" in inference or "replan_every_steps" in inference:
            raise ValueError(
                "inference.action_horizon and replan_every_steps were replaced by model_horizon and execute_steps"
            )
    if float(control.get("frequency_hz", 100)) < float(config.get("camera", {}).get("fps", 30)):
        raise ValueError("Control rate must not be lower than camera rate")
    if (
        config.get("tactile", {}).get("enabled", False)
        and config.get("gripper", {}).get("type") != "dh5"
    ):
        raise ValueError("Tactile images require gripper.type=dh5")
    diagnostics = config.get("diagnostics", {})
    sample_hz = float(diagnostics.get("sample_hz", 0))
    if not math.isfinite(sample_hz) or sample_hz < 0:
        raise ValueError("diagnostics.sample_hz must be finite and nonnegative (0 logs every tick)")
    pending = diagnostics.get("max_pending", 1024)
    if isinstance(pending, bool) or not isinstance(pending, int) or pending <= 0:
        raise ValueError("diagnostics.max_pending must be a positive integer")
