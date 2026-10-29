"""ROS-free snapshot contract shared by the Docker monitor and the web backend."""

import json
import os
import time
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data/runtime/ros2_health.json"


def health_path():
    return Path(os.environ.get("ROS2_HEALTH_PATH", DEFAULT_PATH))


def read_health(path=None, *, now=None):
    now = time.time() if now is None else now
    try:
        payload = json.loads(Path(path or health_path()).read_text())
        age = max(0.0, now - float(payload["updated_at"]))
        if age > 5 and payload["state"] != "disabled":
            payload.update(state="offline", message="ROS2 监控心跳中断，以下为最后收到的状态")
        payload["age_s"] = round(age, 2)
        return payload
    except (OSError, ValueError, KeyError, TypeError):
        return {"state": "not_started", "message": "ROS2 监控未启动或状态尚不可用", "topics": [], "events": []}


def write_health(payload, path=None):
    path = Path(path or health_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)
