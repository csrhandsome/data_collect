"""YAML inheritance without collection, SDK or dataset dependencies."""

import copy
import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(os.environ.get(
    "DATA_COLLECT_CONFIG", Path(__file__).resolve().parents[2] / "config/train/panda.yaml"
))


def load_yaml_config(path=DEFAULT_CONFIG, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError("Cyclic YAML inheritance")
    seen.add(path)
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a YAML mapping")
    parent = payload.pop("extends", None)
    result = load_yaml_config(path.parent / parent, seen) if parent else {}
    merge(result, payload)
    return result


def merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
