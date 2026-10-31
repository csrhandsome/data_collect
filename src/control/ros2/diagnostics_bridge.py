"""Aggregate /diagnostics and publisher discovery for the browser; never open hardware."""

import json
import time
from collections import deque

from control.config import load_config
from control.ros2.health_store import write_health


def expected_topics(config):
    ros = config.get("ros2", {})
    namespace = ros.get("namespace", "").strip("/")

    def relative(topic):
        # Robot topics use the same namespace resolution as Ros2Backend.topic.
        return "/" + "/".join(part for part in (namespace, topic.strip("/")) if part)

    topics = []
    if config.get("robot", {}).get("backend") == "ros2":
        topics.append(relative(ros.get("robot_state_topic", "franka_robot_state_broadcaster/robot_state")))
        if config.get("gripper", {}).get("type", "franka") == "franka":
            topics.append(relative(ros.get("gripper_node", "panda_gripper") + "/joint_states"))
    for section in ("camera", "vr", "audio"):
        cfg = config.get(section, {})
        if cfg.get("backend", config.get("robot", {}).get("backend")) != "ros2":
            continue
        device = cfg.get("ros2", {})
        if section == "camera":
            topics.extend([device.get("external_image_topic", "/camera/external/color/image_raw"),
                           device.get("wrist_image_topic", "/camera/wrist/color/image_raw")])
        elif section == "vr":
            topics.append(device.get("topic", "/vr/input"))
        elif cfg.get("enabled", False):
            topics.append(device.get("topic", "/microphone/audio"))
    return topics


class HealthAggregator:
    def __init__(self, startup_grace_s=10):
        self.started = time.monotonic()
        self.startup_grace_s = startup_grace_s
        self.reports = {}
        self.events = deque(maxlen=100)
        self.previous = {}
        self.event_seq = 0

    def receive(self, message):
        for status in message.status:
            if not status.name.startswith("data_collect/subscription/"):
                continue
            try:
                values = {item.key: json.loads(item.value) for item in status.values}
                if not isinstance(values["topic"], str):
                    continue
                self.reports[status.name] = (time.monotonic(), values)
            except (KeyError, ValueError, TypeError):
                continue

    def snapshot(self, publishers, *, now=None):
        now = time.monotonic() if now is None else now
        topics = []
        for name, (receipt, report) in self.reports.items():
            row = {**report, "id": name}
            if now - receipt > 3 and row["state"] != "stopped":
                row.update(state="warning", reason="subscriber_stale", message="采集订阅心跳中断，可能已退出或阻塞")
            topics.append(row)
        active = {row["topic"] for row in topics if row["state"] != "stopped"}
        for topic, available in publishers.items():
            if topic in active:
                previous = self.previous.get(f"publisher:{topic}")
                if available and previous and previous[0] == "warning":
                    self._transition({
                        "id": f"publisher:{topic}", "topic": topic, "state": "ok",
                        "reason": "recovered", "message": "发布节点已恢复，采集订阅已接入",
                    })
                continue
            warn = not available and now - self.started >= self.startup_grace_s
            topics.append({
                "id": f"publisher:{topic}", "topic": topic, "source": "publisher",
                "component": "驱动发现", "state": "warning" if warn else "waiting",
                "reason": "no_publisher" if warn else "waiting_for_subscriber",
                "message": "未发现发布节点" if warn else (
                    "发布节点在线，等待采集订阅" if available else "等待发布节点启动"
                ),
                "received_hz": None, "min_hz": None, "age_s": None,
            })
        for row in topics:
            self._transition(row)
        warning = any(row["state"] == "warning" for row in topics)
        return {
            "updated_at": time.time(), "state": "warning" if warning else "ok",
            "message": "订阅存在 warning" if warning else "ROS2 监控运行中",
            "topics": topics, "events": list(self.events),
        }

    def _transition(self, row):
        previous = self.previous.get(row["id"])
        current = (row["state"], row["reason"])
        if previous != current and (row["state"] == "warning" or (previous and previous[0] == "warning")):
            self.event_seq += 1
            self.events.appendleft({
                "id": self.event_seq, "time": time.time(), "topic": row["topic"], "state": row["state"],
                "message": row["message"],
            })
        self.previous[row["id"]] = current


def main():
    config = load_config()
    cfg = config.get("ros2", {}).get("monitoring", {})
    if not cfg.get("enabled", False):
        write_health({"updated_at": time.time(), "state": "disabled", "message": "ROS2 监控已关闭", "topics": [], "events": []})
        return
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from rclpy.executors import ExternalShutdownException

    rclpy.init()
    node = rclpy.create_node("data_collect_health", namespace=config.get("ros2", {}).get("namespace", ""))
    aggregator = HealthAggregator(float(cfg.get("startup_grace_s", 10)))
    node.create_subscription(DiagnosticArray, "/diagnostics", aggregator.receive, 100)
    topics = [node.resolve_topic_name(topic) for topic in expected_topics(config)]
    last_event_id = 0

    def update():
        nonlocal last_event_id
        try:
            payload = aggregator.snapshot({topic: node.count_publishers(topic) > 0 for topic in topics})
            for event in reversed(payload["events"]):
                if event["id"] <= last_event_id:
                    continue
                if event["state"] == "warning":
                    node.get_logger().warning(f'{event["topic"]}: {event["message"]}')
                else:
                    node.get_logger().info(f'{event["topic"]}: {event["message"]}')
                last_event_id = event["id"]
            write_health(payload)
        except Exception as exc:
            node.get_logger().warning(f"Health status update unavailable: {exc}", throttle_duration_sec=10)

    node.create_timer(1.0, update)
    update()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
