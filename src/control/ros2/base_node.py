"""ROS device base; Python/direct devices only need control.base_node."""

from abc import abstractmethod

from control.base_node import BaseNode, Sample
from control.ros2.subscription import Subscription


class Ros2Node(BaseNode[Sample]):
    """Own a private ROS runtime and expose decoded samples through ``read``.

    Subclasses keep message types, QoS, validation and startup readiness checks.
    Importing or constructing this base does not import rclpy. ``background=False``
    keeps executor callbacks on the reading thread, as required by the cameras.
    """

    def __init__(self, name, namespace="", *, background=True):
        self._source = Subscription(name, namespace)
        self._background = background

    @abstractmethod
    def _setup(self, node):
        """Register this device's subscriptions on the private rclpy node."""

    def start(self):
        self._source.start(self._setup, background=self._background)

    def close(self):
        self._source.close()
