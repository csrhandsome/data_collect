"""Select one runner for collection, inference and replay."""

from __future__ import annotations

from control.base_runner import BaseRunner


def create_runner(*, config: dict | None = None, backend=None) -> BaseRunner:
    name = (config or {}).get("robot", {}).get("backend", "panda")
    if name == "panda":
        from control.direct.franka_runner_stable import FrankaRunnerStable

        return FrankaRunnerStable(config=config, backend=backend)
    if name == "ros2":
        from control.ros2.franka_runner import FrankaRunner

        return FrankaRunner(config=config, backend=backend)
    raise ValueError("robot.backend must be panda or ros2")
