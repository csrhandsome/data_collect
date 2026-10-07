"""Public Panda API. EE online control and recorded action space are independent."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

from control._panda import lifecycle, motion
from control.gripper_controller import GripperController
from control.robot_state import EEPose
from control.robotic_arm_types import (
    CommandReceipt,
    MotionResult,
    RobotCapabilities,
    RobotState,
    RobotStatus,
)


class RoboticArmControler:
    """Explicit connection, immutable state and one owner for all robot commands."""

    def __init__(self, *, config: dict | None = None, backend=None) -> None:
        lifecycle.initialize(self, config, backend)

    def connect(self) -> RobotStatus:
        return lifecycle.connect(self)

    def close(self) -> None:
        lifecycle.close(self)

    def __enter__(self) -> RoboticArmControler:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception as exc:
                warnings.warn(f"Robot cleanup failed: {exc}", RuntimeWarning)

    def get_state(self) -> RobotState:
        # SDK state reads may call readOnce while idle. Serialize them with
        # controller stop/start in the gripper worker and other state readers.
        with self._lock:
            lifecycle.require_connected(self)
            return self._backend.snapshot(busy=self.gripper_busy)

    def wait_ready(self, *, timeout_s: float = 5.0) -> RobotState:
        from control.util.timing import wait_ready

        return wait_ready(self.get_state, timeout_s)

    def get_status(self) -> RobotStatus:
        return lifecycle.status(self)

    def get_capabilities(self) -> RobotCapabilities:
        return RobotCapabilities(
            gripper=self.gripper.enabled, tactile_images=self.gripper.supports_tactile_images
        )

    def start_stream(self, space: str = "ee") -> None:
        if space != "ee":
            raise ValueError("Online control is EE-only; dataset.action_space selects labels")
        lifecycle.start_stream(self)

    def send_ee_target(
        self, position: Sequence[float], quaternion_xyzw: Sequence[float]
    ) -> CommandReceipt:
        return motion.send_ee(self, position, quaternion_xyzw)

    def hold(self) -> RobotState:
        return motion.hold(self)

    def stop_stream(self, *, timeout_s: float = 2.0) -> None:
        lifecycle.stop_stream(self, timeout_s)

    def move_joints(self, target: Sequence[float], *, speed_factor: float = 0.2) -> MotionResult:
        """Blocking reset/initialization move; unavailable while streaming."""
        return motion.move(self, target, speed_factor=speed_factor)

    def move_ee(
        self,
        position: Sequence[float],
        quaternion_xyzw: Sequence[float],
        *,
        speed_factor: float = 0.2,
    ) -> MotionResult:
        return motion.move(self, position, quaternion_xyzw, speed_factor=speed_factor)

    def move_to_start(self) -> MotionResult:
        from control.util.robot import start_joint_position

        return self.move_joints(start_joint_position(self._config))

    def pose_from_joints(self, joints: Sequence[float]) -> EEPose:
        from control.util.pose import as_vector

        return EEPose.from_matrix(self._backend.pose_from_joints(as_vector(joints, 7, "joints")))

    def execute_ee_trajectory(
        self, points: Sequence[dict], *, on_sample: Callable[[RobotState], None] | None = None
    ) -> MotionResult:
        return motion.trajectory(self, points, on_sample)

    def cancel_motion(self, *, timeout_s: float = 2.0) -> MotionResult:
        return motion.cancel(self, timeout_s)

    def wait_until_stopped(self, *, timeout_s: float = 2.0) -> RobotState:
        return lifecycle.wait_stopped(self, timeout_s)

    @property
    def gripper(self) -> GripperController:
        """The arm-owned common gripper API, selected by gripper.type in YAML."""
        return self._gripper

    @property
    def gripper_busy(self) -> bool:
        return self.gripper.busy

    def set_gripper(
        self,
        open_ratio: float,
        *,
        speed: float = 0.2,
        force: float = 60.0,
        wait: bool = True,
        timeout: float | None = None,
    ) -> CommandReceipt:
        return self.gripper.set_open_ratio(
            open_ratio, speed=speed, force=force, wait=wait, timeout=timeout
        )

    def gripper_open(self, *, wait: bool = True, timeout: float | None = None) -> CommandReceipt:
        return self.gripper.open(wait=wait, timeout=timeout)

    def gripper_close(self, *, wait: bool = True, timeout: float | None = None) -> CommandReceipt:
        return self.gripper.close(wait=wait, timeout=timeout)

    def wait_gripper(self, timeout: float | None = None) -> None:
        self.gripper.wait(timeout)

    def stop_gripper(self) -> None:
        self.gripper.stop()

    def get_tactile_images(self):
        return self.gripper.get_tactile_images()
