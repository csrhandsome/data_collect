"""Public gripper ownership and config-selected SDK adapters, without hardware."""

import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from control._gripper import create_gripper_backend
from control._gripper.dh5 import DH5GripperBackend
from control._gripper.franka import FrankaGripperBackend
from control._panda.backend import PandaBackend
from control._panda.fake import FakeBackend
from control.config import load_config
from control.gripper_controller import GripperController
from control.robotic_arm_controller import RoboticArmControler


@pytest.fixture
def sdk(monkeypatch):
    devices = []

    class FrankaDevice:
        def __init__(self, host):
            self.host, self.width = host, 0.05
            self.calls = []
            self.succeed = True
            devices.append(self)

        def read_once(self):
            return SimpleNamespace(width=self.width)

        def move(self, width, speed):
            self.calls.append(("move", width, speed))
            if self.succeed:
                self.width = width
            return self.succeed

        def grasp(self, *args):
            self.calls.append(("grasp", *args))
            if self.succeed:
                self.width = 0.0
            return self.succeed

        def stop(self):
            self.calls.append(("stop",))

    class DH5Device:
        def __init__(self, **options):
            self.options, self.calls = options, []
            self.succeed = True
            self.observation = SimpleNamespace(
                wrist_img=np.zeros((2, 2, 3), dtype=np.uint8),
                external_img=np.ones((2, 2, 3), dtype=np.uint8),
            )
            devices.append(self)

        def set_force(self, force):
            self.calls.append(("force", force))

        def set_velocity(self, velocity):
            self.calls.append(("velocity", velocity))

        def set_position(self, position, *, wait):
            if not self.succeed:
                raise OSError("serial command failed")
            self.calls.append(("position", position, wait))

        def close(self):
            self.calls.append(("close",))

    raw = SimpleNamespace(
        q=np.zeros(7),
        dq=np.zeros(7),
        O_T_EE=np.eye(4).flatten(),
        time=SimpleNamespace(to_sec=lambda: 0.0),
        robot_mode="idle",
        control_command_success_rate=1.0,
        current_errors="[]",
    )
    monkeypatch.setitem(
        sys.modules,
        "panda_py",
        SimpleNamespace(
            libfranka=SimpleNamespace(Gripper=FrankaDevice),
            Panda=lambda host: SimpleNamespace(get_state=lambda: raw, stop_controller=lambda: None),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "control.soft_gripper_control", SimpleNamespace(DH5Gripper=DH5Device)
    )
    return SimpleNamespace(devices=devices, dh5_class=DH5Device)


@pytest.mark.parametrize(
    "kind,expected", [("franka", FrankaGripperBackend), ("dh5", DH5GripperBackend)]
)
def test_yaml_selection_and_public_interface_use_same_state(sdk, tmp_path, kind, expected):
    path = tmp_path / "robot.yaml"
    path.write_text(
        f"robot:\n  hostname: test-robot\n  activate_fci: false\n"
        f"gripper:\n  type: {kind}\n  open_width_m: 0.08\n  force: 55\n"
        "  velocity: 125\n  dh5:\n    com: test-serial\n"
        f"tactile:\n  enabled: {str(kind == 'dh5').lower()}\n"
    )
    config = load_config(path)
    arm = RoboticArmControler(config=config)
    assert sdk.devices == []  # Construction must not acquire SDK devices.
    assert isinstance(arm.gripper, GripperController)
    assert isinstance(arm._backend.gripper, expected)
    assert arm.gripper.kind == kind and arm.gripper.enabled
    with arm:
        assert arm.gripper.open().accepted
        arm.gripper.set_open_ratio(0.75, wait=False)
        arm.gripper.wait(timeout=1)
        assert arm.gripper.get_state() == arm.get_state().gripper
        assert arm.gripper.get_state().commanded_open_ratio == 0.75
        device = sdk.devices[0]
        if kind == "franka":
            assert device.host == "test-robot"
            assert device.calls[-1] == ("move", 0.06, 0.2)
            assert arm.gripper.get_state().measured_width_m == 0.06
            assert arm.gripper.get_tactile_images() == (None, None)
            arm.gripper.stop()
            assert device.calls[-1] == ("stop",)
        else:
            assert device.options == {"com": "test-serial", "enable_cameras": True}
            assert device.calls[:2] == [("force", 55), ("velocity", 125)]
            assert device.calls[-1] == ("position", 250, True)
            assert arm.gripper.get_state().measured_width_m is None
            left, right = arm.gripper.get_tactile_images()
            assert left is device.observation.wrist_img
            assert right is device.observation.external_img
            with pytest.raises(NotImplementedError, match="stop"):
                arm.gripper.stop()
        arm.gripper.close()
        assert arm.get_state().gripper.commanded_open_ratio == 0
        if kind == "franka":
            assert device.calls[-1] == ("grasp", 0.0, 0.2, 60.0, 0.04, 0.04)
        else:
            assert device.calls[-1] == ("position", 1000, True)
        # Legacy calls must update the same public state and device.
        arm.gripper_open()
        assert arm.gripper.get_state().commanded_open_ratio == 1
    assert arm._backend.gripper.device is None
    if kind == "dh5":
        assert device.calls[-1] == ("close",)


@pytest.mark.parametrize("kind", ["franka", "dh5"])
def test_failed_sdk_command_preserves_last_successful_state(sdk, kind):
    driver = create_gripper_backend({"gripper": {"type": kind}})
    driver.connect()
    try:
        driver.command(0.75, speed=0.2, force=60)
        before = driver.get_state()
        sdk.devices[0].succeed = False
        if kind == "franka":
            assert driver.command(1.0, speed=0.2, force=60) is False
        else:
            with pytest.raises(OSError, match="serial"):
                driver.command(1.0, speed=0.2, force=60)
        assert driver.get_state() == before
    finally:
        driver.close()


def test_dh5_partial_connection_failure_releases_serial_device(sdk, monkeypatch):
    def fail_force(self, force):
        raise OSError("force configuration failed")

    monkeypatch.setattr(sdk.dh5_class, "set_force", fail_force)
    backend = PandaBackend({"robot": {"activate_fci": False}, "gripper": {"type": "dh5"}})
    with pytest.raises(OSError, match="force configuration"):
        backend.connect()
    assert sdk.devices[0].calls == [("close",)]
    assert backend.gripper.device is None and backend.panda is None
    backend.close()


def test_public_and_legacy_commands_share_worker_and_stream_ownership():
    backend = FakeBackend()
    release = threading.Event()
    command = backend.gripper_command

    def blocked(*args, **kwargs):
        if not release.wait(2):
            raise TimeoutError("Command was not released")
        return command(*args, **kwargs)

    backend.gripper_command = blocked
    with RoboticArmControler(backend=backend) as arm:
        arm.start_stream()
        try:
            arm.gripper.close(wait=False)
            assert arm.gripper.busy and arm.gripper_busy
            with pytest.raises(RuntimeError, match="busy"):
                arm.gripper_open(wait=False)
            with pytest.raises(TimeoutError):
                arm.gripper.wait(timeout=0.001)
        finally:
            release.set()
        arm.wait_gripper(timeout=1)
        assert not arm.gripper.busy
        assert arm.gripper.get_state().commanded_open_ratio == 0
        assert backend.streaming and backend.calls.count("stop") == 1
        assert backend.calls.count("start") == 2


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan")])
def test_invalid_public_command_does_not_start_worker(ratio):
    with RoboticArmControler(backend=FakeBackend()) as arm:
        with pytest.raises(ValueError, match="ratio"):
            arm.gripper.set_open_ratio(ratio)
        assert arm._gripper_executor is None


def test_disconnected_and_disabled_public_gripper_reject_commands():
    arm = RoboticArmControler(backend=FakeBackend())
    for call in [
        arm.gripper.get_state,
        arm.gripper.open,
        arm.gripper.stop,
        arm.gripper.get_tactile_images,
    ]:
        with pytest.raises(RuntimeError, match="disconnected"):
            call()
    with RoboticArmControler(config={"gripper": {"type": "none"}}, backend=FakeBackend()) as arm:
        arm.start_stream()
        assert not arm.gripper.enabled and not arm.get_capabilities().gripper
        with pytest.raises(RuntimeError, match="disabled"):
            arm.gripper.open()
        assert arm._gripper_executor is None and arm._backend.streaming


def test_unknown_type_rejected_before_sdk_connection(sdk):
    with pytest.raises(ValueError, match="Unknown gripper"):
        RoboticArmControler(config={"gripper": {"type": "unknown"}})
    assert sdk.devices == []
