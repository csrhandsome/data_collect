"""Hardware contract independent of the Panda arm SDK and command worker."""

from abc import ABC, abstractmethod

from control.robotic_arm_types import GripperState


class GripperBackend(ABC):
    kind: str

    def __init__(self, config: dict) -> None:
        self.config = config.get("gripper", {})
        self.commanded_open_ratio = 1.0
        self.measured_width_m = None
        self.device = None

    @abstractmethod
    def connect(self) -> None:
        """Acquire the device; SDK imports must be lazy."""

    @abstractmethod
    def close(self) -> None:
        """Release the device, including after a partial connect failure."""

    @abstractmethod
    def command(self, ratio: float, *, speed: float, force: float) -> bool:
        """Execute one blocking command and update state only on success."""

    @abstractmethod
    def stop(self) -> None:
        """Stop motion or explicitly report that stopping is unsupported."""

    def get_state(self, *, busy: bool = False) -> GripperState:
        return GripperState(self.kind, self.commanded_open_ratio, self.measured_width_m, busy)

    def tactile_images(self):
        return None, None

    def tactile_frames(self):
        return None, None, None, None

    def require_connected(self) -> None:
        if self.device is None:
            raise RuntimeError("Gripper is disconnected; connect the arm first")


class DisabledGripperBackend(GripperBackend):
    kind = "none"

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def command(self, ratio: float, *, speed: float, force: float) -> bool:
        raise RuntimeError("Gripper is disabled")

    def stop(self) -> None:
        pass
