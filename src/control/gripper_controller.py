"""Public gripper API owned by a robotic arm controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

from control._panda import gripper, lifecycle
from control.robotic_arm_types import CommandReceipt, GripperState

if TYPE_CHECKING:
    from control.robotic_arm_controller import RoboticArmControler


class GripperController:
    """Common Franka/DH5 interface; connection and command scheduling belong to the arm.

    Open ratios use 0 for closed and 1 for fully open. Franka keeps its existing
    move/grasp behavior; DH5 maps the ratio to its 0..1000 position register.
    """

    def __init__(self, arm: RoboticArmControler) -> None:
        self._arm = arm

    @property
    def kind(self) -> str:
        return self._arm._config.get("gripper", {}).get("type", "franka")

    @property
    def enabled(self) -> bool:
        return self.kind != "none"

    @property
    def supports_tactile_images(self) -> bool:
        return self.kind == "dh5"

    @property
    def busy(self) -> bool:
        return lifecycle.status(self._arm).gripper_busy

    def get_state(self) -> GripperState:
        return self._arm.get_state().gripper

    def set_open_ratio(
        self,
        open_ratio: float,
        *,
        speed: float = 0.2,
        force: float = 60.0,
        wait: bool = True,
        timeout: float | None = None,
    ) -> CommandReceipt:
        """Submit a command through the arm's single worker, optionally waiting.

        speed (m/s) and force (N) apply to Franka. DH5 uses the native velocity
        and force values in YAML, as in the existing collection implementation.
        """
        return gripper.command(self._arm, open_ratio, speed, force, wait, timeout)

    def open(self, *, wait: bool = True, timeout: float | None = None) -> CommandReceipt:
        return self.set_open_ratio(1.0, wait=wait, timeout=timeout)

    def close(self, *, wait: bool = True, timeout: float | None = None) -> CommandReceipt:
        """Close the jaws. Resource cleanup is handled by arm.close()."""
        return self.set_open_ratio(0.0, wait=wait, timeout=timeout)

    def wait(self, timeout: float | None = None) -> None:
        gripper.wait(self._arm, timeout)

    def stop(self) -> None:
        """Stop Franka motion; DH5's driver has no confirmed stop command."""
        lifecycle.require_connected(self._arm)
        self._arm._backend.stop_gripper()

    def get_tactile_images(self):
        lifecycle.require_connected(self._arm)
        return self._arm._backend.tactile_images()

    def get_tactile_frames(self):
        """Left/right RGB and their host capture timestamps, read atomically."""
        lifecycle.require_connected(self._arm)
        return self._arm._backend.tactile_frames()
