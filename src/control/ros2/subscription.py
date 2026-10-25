"""Private ROS contexts for device subscribers; ROS imports stay lazy."""

import threading
import time
from uuid import uuid4


class Subscription:
    def __init__(self, name, namespace=""):
        self.name, self.namespace = name, namespace
        self.condition = threading.Condition()
        self.node = self.context = self.executor = self.thread = None
        self.failure = None

    def start(self, setup):
        from rclpy import init
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.signals import SignalHandlerOptions

        try:
            self.failure = None
            self.context = Context()
            init(args=[], context=self.context, signal_handler_options=SignalHandlerOptions.NO)
            self.node = Node(
                f"{self.name}_{uuid4().hex[:8]}", namespace=self.namespace or "/",
                context=self.context, use_global_arguments=False,
            )
            setup(self.node)
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self._spin, name=self.name, daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise

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
        with self.condition:
            while not predicate():
                self.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"No valid {self.name} data; check ROS publishers and topics")
                self.condition.wait(remaining)
            self.check()

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=2)
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError(f"{self.name} worker did not stop; resources retained")
        if self.node is not None:
            self.node.destroy_node()
        if self.context is not None:
            self.context.try_shutdown()
            self.context.destroy()
        self.node = self.context = self.executor = self.thread = None
