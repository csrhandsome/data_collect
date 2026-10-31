"""Warning transitions, shared snapshots, and opt-in real subscription diagnostics."""

import json
import os
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from control.ros2.diagnostics import StreamStatistics
from control.ros2.diagnostics_bridge import HealthAggregator, expected_topics
from control.ros2.health_store import read_health, write_health


def diagnostic(topic="/camera", state="ok", reason="ok"):
    values = dict(topic=topic, state=state, reason=reason, message=reason, component="camera", source="subscriber")
    return SimpleNamespace(status=[SimpleNamespace(
        name=f"data_collect/subscription/camera/{topic}",
        values=[SimpleNamespace(key=key, value=json.dumps(value)) for key, value in values.items()],
    )])


def test_statistics_startup_timeout_low_rate_invalid_and_recovery():
    stream = StreamStatistics("/camera", min_hz=5, timeout_s=1, window_s=2, startup_grace_s=1, started=0)
    assert stream.snapshot(0.5)["state"] == "waiting"
    assert stream.snapshot(1.1)["reason"] == "no_messages"
    stream.tick(now=2.1)
    assert stream.snapshot(2.1)["reason"] == "low_rate"
    for i in range(20):
        stream.tick(now=2.2 + i * 0.1)
    assert stream.snapshot(4.1)["state"] == "ok"
    stream.tick(False, now=4.2)
    assert stream.snapshot(4.2)["reason"] == "invalid"
    stream.tick(now=4.3)
    assert stream.snapshot(4.3)["state"] == "ok"
    assert stream.snapshot(5.5)["reason"] == "timeout"
    assert stream.invalid == 1


def test_aggregator_distinguishes_discovery_actual_subscriptions_and_stalled_executor(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 0)
    aggregator = HealthAggregator(startup_grace_s=2)
    assert aggregator.snapshot({"/camera": True}, now=1)["topics"][0]["state"] == "waiting"
    missing = aggregator.snapshot({"/camera": False}, now=3)
    assert missing["state"] == "warning"
    assert len(missing["events"]) == 1
    assert len(aggregator.snapshot({"/camera": False}, now=4)["events"]) == 1
    aggregator.receive(diagnostic())
    report = aggregator.snapshot({"/camera": True}, now=0)
    assert len(report["topics"]) == 1
    assert report["topics"][0]["source"] == "subscriber"
    assert report["events"][0]["state"] == "ok"
    stale = aggregator.snapshot({"/camera": True}, now=4)
    assert stale["topics"][0]["reason"] == "subscriber_stale"
    aggregator.receive(diagnostic())
    recovered = aggregator.snapshot({"/camera": True}, now=0)
    assert recovered["state"] == "ok"
    assert recovered["events"][0]["state"] == "ok"


def test_snapshot_never_claims_healthy_when_monitor_is_gone(tmp_path):
    path = tmp_path / "health.json"
    assert read_health(path)["state"] == "not_started"
    write_health(dict(state="ok", updated_at=100, topics=[], events=[]), path)
    assert read_health(path, now=101)["state"] == "ok"
    assert read_health(path, now=106)["state"] == "offline"
    path.write_text("broken")
    assert read_health(path)["state"] == "not_started"


def test_monitor_respects_enabled_devices_and_robot_namespace():
    config = {
        "robot": {"backend": "ros2"}, "ros2": {"namespace": "arm"},
        "camera": {"backend": "direct"}, "gripper": {"type": "none"},
        "vr": {"ros2": {"topic": "/vr/input"}}, "audio": {"enabled": False},
    }
    assert expected_topics(config) == ["/arm/franka_robot_state_broadcaster/robot_state", "/vr/input"]


@pytest.mark.skipif(os.environ.get("DATA_COLLECT_TEST_ROS2") != "1", reason="Requires ROS 2 and DDS")
def test_actual_subscription_diagnostics_warn_and_recover_over_dds():
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import String

    from control.ros2.subscription import Subscription

    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node("health_test_peer", context=context)
    topic = "/health_test_" + uuid4().hex[:8]
    publisher = node.create_publisher(String, topic, 1)
    aggregator = HealthAggregator(0.1)
    node.create_subscription(DiagnosticArray, "/diagnostics", aggregator.receive, 10)
    active = [True]
    node.create_timer(0.02, lambda: publisher.publish(String(data="valid")) if active[0] else None)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    source = Subscription("health_test", monitoring={"enabled": True, "startup_grace_s": 0.1, "window_s": 1})
    received = []

    def until(predicate):
        deadline = time.monotonic() + 6
        while not predicate():
            assert time.monotonic() < deadline, "Timed out waiting for diagnostics"
            time.sleep(0.02)

    def state():
        rows = aggregator.snapshot({})["topics"]
        return next((row for row in rows if row["topic"] == topic), {})

    try:
        source.start(lambda _: source.subscribe(String, topic, received.append, 1, expected_hz=10, timeout_s=0.3))
        until(lambda: state().get("state") == "ok")
        assert received
        active[0] = False
        until(lambda: state().get("reason") == "timeout")
        assert source.failure is None
        active[0] = True
        until(lambda: state().get("state") == "ok")
        assert any(event["state"] == "ok" for event in aggregator.events)
    finally:
        source.close()
        executor.shutdown(timeout_sec=2)
        thread.join(timeout=2)
        node.destroy_node()
        context.try_shutdown()
        context.destroy()
