"""Transport-independent latest-sample interface for acquisition devices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

Sample = TypeVar("Sample")


class BaseNode(ABC, Generic[Sample]):
    """Read the latest device sample, without introducing a queue or ROS dependency.

    Factories may return already-started nodes. ``start`` must then be a no-op.
    Reads preserve device-specific freshness checks and may wait for valid data;
    they do not promise synchronization between different devices. Returned data
    should be treated as read-only. Robot commands belong to ``BaseRunner``.
    """

    @abstractmethod
    def start(self) -> None:
        """Connect and prepare input. This never starts robot motion."""

    @abstractmethod
    def read(self) -> Sample:
        """Return the latest sample, raising if the device cannot supply it."""

    @abstractmethod
    def close(self) -> None:
        """Release resources. Repeated calls must be safe."""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
