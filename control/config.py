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
        "DATA_COLLECT_CONFIG", Path(__file__).resolve().parents[1] / "config/train/panda.yaml"
    )
)


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
        validate(result)
    return result


def merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def validate(config):
    control = config.get("control", {})
    if control.get("mode", "ee") != "ee":
        raise ValueError("control.mode must be ee; action fields select saved joint/ee labels")
    if config.get("dataset", {}).get("action_space", "joint") not in ("joint", "ee"):
        raise ValueError("dataset.action_space must be joint or ee")
    if "fields" in config.get("dataset", {}):
        raise ValueError("dataset.fields was replaced by observation/action mappings")
    features(
        int(config.get("camera", {}).get("image_hw", 224)),
        config.get("dataset", {}).get("action_space", "joint"),
        config.get("tactile", {}).get("enabled", False),
        config.get("observation"),
        config.get("action"),
    )
    if config.get("gripper", {}).get("type", "franka") not in ("franka", "dh5", "none"):
        raise ValueError("gripper.type must be franka, dh5 or none")
    for section, key, default in [
        ("control", "frequency_hz", 100),
        ("camera", "fps", 30),
        ("inference", "action_fps", 30),
        ("audio", "sample_rate", 16000),
    ]:
        positive_rate(config.get(section, {}).get(key, default), f"{section}.{key}")
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
