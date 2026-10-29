import time

from fastapi.testclient import TestClient

from control.ros2.health_store import write_health
from replay.backend.main import create_app


def test_ros2_health_api_reads_shared_status_and_detects_monitor_exit(tmp_path, monkeypatch):
    path = tmp_path / "ros2_health.json"
    monkeypatch.setenv("ROS2_HEALTH_PATH", str(path))
    with TestClient(create_app(data_root=tmp_path)) as client:
        assert client.get("/api/ros2/health").json()["state"] == "not_started"
        payload = {
            "state": "warning", "updated_at": time.time(), "message": "订阅存在 warning",
            "topics": [{"topic": "/camera", "state": "warning", "message": "接收频率偏低"}],
            "events": [],
        }
        write_health(payload, path)
        response = client.get("/api/ros2/health")
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["topics"] == payload["topics"]
        payload["updated_at"] -= 10
        write_health(payload, path)
        assert client.get("/api/ros2/health").json()["state"] == "offline"
