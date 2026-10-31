"""Warning-only health checks on the application's actual subscription callbacks."""

import json
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class StreamStatistics:
    topic: str
    min_hz: float = 0.0
    timeout_s: float = 1.0
    window_s: float = 5.0
    startup_grace_s: float = 10.0
    started: float = field(default_factory=time.monotonic)
    received: int = 0
    invalid: int = 0
    last_received: float | None = None
    last_valid: float | None = None
    _samples: deque = field(default_factory=deque)
    _reported_invalid: int = 0
    source_age_s: float | None = None

    def tick(self, valid=True, *, now=None, source_age_s=None):
        now = time.monotonic() if now is None else now
        self.received += 1
        self.last_received = now
        self.source_age_s = source_age_s
        if valid:
            self.last_valid = now
        else:
            self.invalid += 1

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        if not self._samples:
            self._samples.append((self.started, 0))
        while len(self._samples) > 1 and self._samples[1][0] <= now - self.window_s:
            self._samples.popleft()
        since, count = self._samples[0]
        duration = now - since
        hz = (self.received - count) / duration if duration > 0 else 0.0
        self._samples.append((now, self.received))
        age = None if self.last_received is None else max(0.0, now - self.last_received)
        valid_age = None if self.last_valid is None else max(0.0, now - self.last_valid)
        state, reason, message = "ok", "ok", "接收正常"
        if now - self.started < self.startup_grace_s:
            state, reason, message = "waiting", "starting", "订阅启动中"
        elif self.last_received is None:
            state, reason, message = "warning", "no_messages", "尚未收到消息，请检查发布节点和 QoS"
        elif age > self.timeout_s:
            state, reason, message = "warning", "timeout", f"消息中断，已 {age:.2f} 秒未收到数据"
        elif self.invalid > self._reported_invalid or valid_age is None or valid_age > self.timeout_s:
            state, reason, message = "warning", "invalid", "收到无效消息，请检查数据内容或源时间戳"
        elif duration >= self.window_s and hz < self.min_hz:
            state, reason = "warning", "low_rate"
            message = f"接收频率偏低：{hz:.1f} Hz，最低 {self.min_hz:.1f} Hz；可能跳帧或处理滞后"
        self._reported_invalid = self.invalid
        return {
            "topic": self.topic, "state": state, "reason": reason, "message": message,
            "received_hz": round(hz, 2), "min_hz": self.min_hz,
            "age_s": None if age is None else round(age, 3),
            "received": self.received, "invalid": self.invalid,
            "source_age_s": self.source_age_s,
        }


class SubscriptionDiagnostics:
    """Publish tiny DiagnosticArray reports, without subscribing to sensor streams twice."""

    def __init__(self, node, component, config):
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

        self.node, self.component, self.config = node, component, config
        self._array_type, self._status_type, self._value_type = DiagnosticArray, DiagnosticStatus, KeyValue
        self.streams = []
        self.publisher = node.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.timer = node.create_timer(1.0, self.publish)

    def subscribe(self, message_type, topic, callback, qos, *, min_hz=0.0, timeout_s=1.0):
        stream = StreamStatistics(
            self.node.resolve_topic_name(topic), min_hz, timeout_s,
            float(self.config.get("window_s", 5)), float(self.config.get("startup_grace_s", 10)),
        )
        self.streams.append(stream)

        def receive(message):
            try:
                result = callback(message)
            except Exception:
                stream.tick(False)
                raise
            source_age = None
            header = getattr(message, "header", None)
            if header is not None:
                stamp_ns = header.stamp.sec * 1_000_000_000 + header.stamp.nanosec
                source_age = (self.node.get_clock().now().nanoseconds - stamp_ns) / 1e9
            valid = result is not False and (source_age is None or -0.1 <= source_age <= timeout_s)
            stream.tick(valid, source_age_s=source_age)

        return self.node.create_subscription(message_type, topic, receive, qos)

    def publish(self, *, stopped=False):
        # Diagnostics must never change control or recording behavior.
        try:
            report = self._array_type()
            report.header.stamp = self.node.get_clock().now().to_msg()
            for stream in self.streams:
                values = stream.snapshot()
                if stopped:
                    values.update(state="stopped", reason="stopped", message="订阅已停止")
                status = self._status_type()
                status.name = f"data_collect/subscription/{self.component}/{stream.topic}"
                status.hardware_id = "data_collect"
                status.level = (
                    self._status_type.WARN if values["state"] == "warning" else self._status_type.OK
                )
                status.message = values["message"]
                status.values = [
                    self._value_type(key=key, value=json.dumps(value, ensure_ascii=False))
                    for key, value in {**values, "source": "subscriber", "component": self.component}.items()
                ]
                report.status.append(status)
            self.publisher.publish(report)
        except Exception as exc:
            self.node.get_logger().warning(f"Subscription diagnostics unavailable: {exc}", throttle_duration_sec=10)
