"""Franka parallel gripper implementation, preserving collection behavior."""

from control._gripper.backend import GripperBackend


class FrankaGripperBackend(GripperBackend):
    kind = "franka"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.hostname = config.get("robot", {}).get("hostname", "192.168.1.100")

    def connect(self) -> None:
        from panda_py import libfranka

        if self.device is None:
            try:
                self.device = libfranka.Gripper(self.hostname)
                self.measured_width_m = float(self.device.read_once().width)
                self.commanded_open_ratio = 1.0
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        # libfranka.Gripper releases its connection when the SDK object is released.
        self.device = None
        self.measured_width_m = None

    def command(self, ratio: float, *, speed: float, force: float) -> bool:
        self.require_connected()
        if ratio >= 0.5:
            width = float(self.config.get("open_width_m", 0.05)) * ratio
            result = self.device.move(width, speed)
        else:
            result = self.device.grasp(0.0, speed, force, 0.04, 0.04)
        if result:
            self.commanded_open_ratio = ratio
            self.measured_width_m = float(self.device.read_once().width)
        return bool(result)

    def stop(self) -> None:
        self.require_connected()
        self.device.stop()
