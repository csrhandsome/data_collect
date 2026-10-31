"""Common reader contract and ROS execution ownership without hardware or DDS."""

import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from control.base_node import BaseNode
from control.collection.devices import make_cameras, make_vr
from control.ros2.base_node import Ros2Node
from control.ros2.camera_connector import Ros2Cameras
from control.ros2.subscription import Subscription
from control.types import CameraPair


def test_dry_run_devices_share_reader_contract_and_legacy_interfaces():
    config = {"camera": {"image_hw": 8}}
    with make_cameras(config, dry_run=True) as cameras, make_vr(config, dry_run=True) as vr:
        assert isinstance(cameras, BaseNode)
        assert isinstance(vr, BaseNode)
        pair = cameras.read()
        assert isinstance(pair, CameraPair)
        assert pair.front.shape == cameras.get_frames().front.shape == (8, 8, 3)
        assert vr.read().arm_enabled == vr.latest.arm_enabled


def test_node_imports_and_runner_construction_need_no_hardware_sdks():
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'rclpy', 'panda_py', 'sounddevice', 'pyrealsense2', 'teleop_xr'}:
        raise AssertionError('Unexpected SDK import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from control.base_node import BaseNode
from control.ros2.base_node import Ros2Node
from control.ros2.camera_connector import Ros2Cameras
from control.ros2.vr_connector import Ros2VR
from control.ros2.microphone_connector import Ros2Microphone
from control.runners import create_runner
assert issubclass(Ros2Cameras, Ros2Node)
assert issubclass(Ros2VR, Ros2Node)
for name in ('panda', 'ros2'):
    runner = create_runner(config={'robot': {'backend': name}})
    assert isinstance(runner, BaseNode)
    runner.close()
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


def test_ros_base_requires_device_read_and_setup():
    with pytest.raises(TypeError):
        BaseNode()
    with pytest.raises(TypeError):
        Ros2Node("incomplete")


def test_manual_ros_wait_pumps_callbacks_and_does_not_create_spin_thread():
    source = Subscription("camera")
    source._background = False
    source.node = Mock()
    received = []
    source.executor = Mock()
    source.executor.spin_once.side_effect = lambda **kwargs: received.append("frame")
    source.wait(lambda: bool(received), 0.1)
    assert received
    assert source.thread is None
    # Starting a prepared source must preserve subscriptions and the executor.
    setup = Mock()
    source.start(setup)
    setup.assert_not_called()
    source.close()


@pytest.fixture
def ros_runtime(monkeypatch):
    runtime = SimpleNamespace(context=Mock(), node=Mock(), executor=Mock(), init=Mock())
    modules = {
        "rclpy": SimpleNamespace(init=runtime.init),
        "rclpy.context": SimpleNamespace(Context=lambda: runtime.context),
        "rclpy.node": SimpleNamespace(Node=Mock(return_value=runtime.node)),
        "rclpy.executors": SimpleNamespace(SingleThreadedExecutor=lambda **kwargs: runtime.executor),
        "rclpy.signals": SimpleNamespace(SignalHandlerOptions=SimpleNamespace(NO="no_signals")),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return runtime


def test_ros_start_owns_one_private_context_and_cleans_partial_setup(ros_runtime):
    source = Subscription("camera", "/test")
    setup = Mock(side_effect=ValueError("bad topic"))
    with pytest.raises(ValueError, match="bad topic"):
        source.start(setup, background=False)
    setup.assert_called_once_with(ros_runtime.node)
    ros_runtime.init.assert_called_once_with(
        args=[], context=ros_runtime.context, signal_handler_options="no_signals",
    )
    ros_runtime.node.destroy_node.assert_called_once_with()
    ros_runtime.context.try_shutdown.assert_called_once_with()
    ros_runtime.context.destroy.assert_called_once_with()
    assert source.node is source.context is source.executor is source.thread is None


def test_ros_node_context_starts_once_and_keeps_manual_execution(ros_runtime):
    class Input(Ros2Node[int]):
        def _setup(self, node):
            self.sample = 42

        def read(self):
            return self.sample

    device = Input("test", background=False)
    with device:
        device.start()
        assert device.read() == 42
        assert device._source.thread is None
        ros_runtime.executor.add_node.assert_called_once_with(ros_runtime.node)
        assert ros_runtime.init.call_count == 1
    ros_runtime.executor.shutdown.assert_called_once_with(timeout_sec=2)
    ros_runtime.node.destroy_node.assert_called_once_with()


def test_background_ros_executor_cannot_also_be_polled():
    source = Subscription("vr")
    source.executor = Mock()
    with pytest.raises(RuntimeError, match="background"):
        source.poll(0)
    source.executor.spin_once.assert_not_called()


def test_executor_failure_reaches_readiness_wait_even_with_cached_data():
    source = Subscription("robot")
    error = ValueError("callback failed")
    source.failure = error
    with pytest.raises(RuntimeError, match="subscriber failed") as caught:
        source.wait(lambda: True, 0.1)
    assert caught.value.__cause__ is error


def test_cleanup_stops_callbacks_before_destroying_action_clients_and_context():
    source = Subscription("robot")
    events = []
    source.executor = Mock(shutdown=lambda **kwargs: events.append("executor"))
    source.thread = Mock(join=lambda **kwargs: events.append("join"), is_alive=lambda: False)
    source.node = Mock(destroy_node=lambda: events.append("node"))
    source.context = Mock(
        try_shutdown=lambda: events.append("shutdown"), destroy=lambda: events.append("context"),
    )
    source.close(before_destroy=lambda: events.append("actions"))
    source.close()
    assert events == ["executor", "join", "actions", "node", "shutdown", "context"]
    assert source.node is source.executor is source.thread is source.context is None


def test_live_worker_retains_ros_resources():
    source = Subscription("robot")
    source.executor, source.node, source.context = Mock(), Mock(), Mock()
    source.thread = Mock(is_alive=lambda: True)
    cleanup = Mock()
    with pytest.raises(RuntimeError, match="resources retained"):
        source.close(before_destroy=cleanup)
    cleanup.assert_not_called()
    source.node.destroy_node.assert_not_called()
    source.context.destroy.assert_not_called()


def test_camera_read_services_pending_callbacks_even_when_cache_is_fresh():
    cameras = Ros2Cameras.__new__(Ros2Cameras)
    cameras._source = Mock()
    cameras._max_age = 0.2
    now = time.monotonic_ns()
    cameras._raw = {key: (b"", 1, 1, 3, "rgb8", now, 1) for key in ("front", "wrist")}
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    cameras._resized = Mock(return_value=(image, now, 1))
    assert cameras.read().front is image
    cameras._source.poll.assert_called_once_with(0.0)


def test_direct_vr_keeps_process_boundary_and_idempotent_start():
    from control.vr_input import VRInputProcess

    vr = VRInputProcess.__new__(VRInputProcess)
    vr._proc = SimpleNamespace(pid=None, start=Mock(), is_alive=lambda: True, terminate=Mock(), join=Mock())
    vr.start()
    vr._proc.pid = 123
    vr.start()
    vr._proc.start.assert_called_once_with()
    vr.close()
    vr._proc.terminate.assert_called_once_with()
    vr._proc.join.assert_called_once_with(timeout=2.0)
