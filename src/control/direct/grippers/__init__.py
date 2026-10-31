"""Config-selected hardware implementations behind the public gripper API."""

from control.direct.grippers.backend import DisabledGripperBackend, GripperBackend
from control.direct.grippers.dh5 import DH5GripperBackend
from control.direct.grippers.franka import FrankaGripperBackend


def create_gripper_backend(config: dict) -> GripperBackend:
    kind = config.get("gripper", {}).get("type", "franka")
    implementations = {
        "franka": FrankaGripperBackend,
        "dh5": DH5GripperBackend,
        "none": DisabledGripperBackend,
    }
    if kind not in implementations:
        raise ValueError(f"Unknown gripper: {kind}")
    return implementations[kind](config)
