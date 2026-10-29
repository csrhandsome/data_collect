"""Private ROS contexts for device subscribers; ROS imports stay lazy."""

import threading
import time
from uuid import uuid4


class Subscription:
    def __init__(self, name, namespace="", *, monitoring=None):
        self.name, self.namespace = name, namespace
        self.condition = threading.Condition()
        self.node = self.context = self.executor = self.thread = None
        self.failure = None
        self._background = True
        self.monitoring = monitoring or {}
        self._diagnostics = None

    def start(self, setup, *, background=True):
        if self.node is not None:
            self.check()
            return
        from rclpy import init
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.signals import SignalHandlerOptions

        try:
            self.failure = None
            self._background = background
            self.context = Context()
            init(args=[], context=self.context, signal_handler_options=SignalHandlerOptions.NO)
            self.node = Node(
                f"{self.name}_{uuid4().hex[:8]}", namespace=self.namespace or "/",
                context=self.context, use_global_arguments=False,
            )
            if self.monitoring.get("enabled", False):
                try:
                    from control.ros2.diagnostics import SubscriptionDiagnostics

                    self._diagnostics = SubscriptionDiagnostics(self.node, self.name, self.monitoring)
                except Exception as exc:
                    self.node.get_logger().warning(f"Subscription monitoring unavailable: {exc}")
            setup(self.node)
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            if background:
                self.thread = threading.Thread(target=self._spin, name=self.name, daemon=True)
                self.thread.start()
        except BaseException:
            self.close()
            raise

    def subscribe(self, message_type, topic, callback, qos, *, expected_hz=0.0, timeout_s=1.0):
        if self._diagnostics is None:
            return self.node.create_subscription(message_type, topic, callback, qos)
        return self._diagnostics.subscribe(
            message_type, topic, callback, qos,
            min_hz=expected_hz * float(self.monitoring.get("min_rate_ratio", 0.8)),
            timeout_s=timeout_s,
        )

    def poll(self, timeout_sec, *, limit=8):
        """Handle pending callbacks on the caller thread. No background spin required."""
        self.check()
        if self.executor is None:
            return
        if self._background:
            raise RuntimeError("Cannot poll a ROS executor that is spinning in the background")
        # One spin_once serves a single subscription. Take a few so both cameras
        # move forward, without resizing a long backlog on this thread.
        self.executor.spin_once(timeout_sec=max(0.0, float(timeout_sec)))
        for _ in range(max(0, int(limit) - 1)):
            self.executor.spin_once(timeout_sec=0.0)
        self.check()

    def _spin(self):
        try:
            self.executor.spin()
        except Exception as exc:
            with self.condition:
                self.failure = exc
                self.condition.notify_all()

    def check(self):
        if self.failure is not None:
            raise RuntimeError(f"{self.name} subscriber failed") from self.failure

    def wait(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        if not self._background:
            while not predicate():
                self.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"No valid {self.name} data; check ROS publishers and topics")
                self.poll(min(0.05, remaining))
            self.check()
            return
        with self.condition:
            while not predicate():
                self.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"No valid {self.name} data; check ROS publishers and topics")
                self.condition.wait(remaining)
            self.check()

    def close(self, *, before_destroy=None):
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=2)
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError(f"{self.name} worker did not stop; resources retained")
        if self._diagnostics is not None:
            self._diagnostics.publish(stopped=True)
            self._diagnostics = None
        if before_destroy is not None:
            before_destroy()
        if self.node is not None:
            self.node.destroy_node()
        if self.context is not None:
            self.context.try_shutdown()
            self.context.destroy()
        self.node = self.context = self.executor = self.thread = None
