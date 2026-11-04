"""Opt-in official UR mock hardware integration; no robot, cameras or network needed.

DATA_COLLECT_TEST_UR=1 python -m pytest -q src/tests/ros2/test_ur_live.py
Run inside the project image with --network none and an independent ROS_DOMAIN_ID.
"""

import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from control.config import load_config
from control.runners import create_runner
from control.util.pose import matrix_to_quat_xyzw
from env.bringup_ur import launch_command

pytestmark = pytest.mark.skipif(os.environ.get("DATA_COLLECT_TEST_UR") != "1",
                                reason="Opt-in UR ROS2 mock hardware test")


@pytest.fixture(params=["ur5", "ur5e"])
def deployment(request, tmp_path):
    config = load_config(Path(__file__).resolve().parents[3] / "config/train/ur.yaml")
    config["robot"]["type"] = request.param
    config["ros2"]["use_mock_hardware"] = True
    config["ros2"]["monitoring"]["enabled"] = False
    config["ros2"]["connect_timeout_s"] = 30
    logfile = tmp_path / "ur-driver.log"
    with logfile.open("w") as stream:
        process = subprocess.Popen(launch_command(config), stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            yield config, process, logfile
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)


def test_mock_state_stream_hold_and_collection(deployment, tmp_path, monkeypatch):
    from control.collection import pipeline
    from control.collection.devices import FakeCameras, FakeVR
    from replay.scripts.read_dataset import read_dataset

    config, driver, logfile = deployment
    arm = create_runner(config=config)
    try:
        arm.connect()
        assert driver.poll() is None, logfile.read_text()
        state = arm.read()
        assert state.base_frame == "base_link" and state.joint_positions.shape == (6,)
        from sensor_msgs.msg import JointState

        from control.ros2.bridge import Ros2Publisher
        from control.ros2.subscription import Subscription

        observer, received = Subscription("ur_telemetry_observer"), []
        try:
            observer.start(lambda node: node.create_subscription(
                JointState, "/data_collect/robot/joint_states", received.append, 1))
            with Ros2Publisher(config) as bridge:
                deadline = time.monotonic() + 5
                while not received and time.monotonic() < deadline:
                    bridge.submit(arm.read())
                    time.sleep(0.02)
                assert received and received[-1].name == config["ros2"]["joint_names"]
                assert len(received[-1].position) == 6
        finally:
            observer.close()
        # Move away from singular straight-arm mock defaults using the same
        # bounded joint API. Only the driver's GenericSystem is connected.
        target_joints = state.joint_positions.copy()
        target_joints[2] += 0.3
        target_joints[4] += 0.3
        arm.move_joints(target_joints, speed_factor=1)
        state = arm.read()
        desired = arm._backend.pose_from_joints(state.joint_positions + 0.003)
        quaternion = matrix_to_quat_xyzw(desired[:3, :3])
        arm.start_stream()
        before = state.joint_positions.copy()
        for _ in range(40):
            arm.send_ee_target(desired[:3, 3], quaternion)
            time.sleep(0.01)
        assert np.max(np.abs(arm.read().joint_positions - before)) > 0.0001
        arm.hold()
        arm.stop_stream()
        stopped = arm.read().joint_positions.copy()
        time.sleep(0.25)
        np.testing.assert_allclose(arm.read().joint_positions, stopped, atol=0.005)
    except Exception:
        print(logfile.read_text())
        raise
    finally:
        arm.close()

    # Real UR runner and driver; only the two cameras and VR inputs are synthetic.
    config["ros2"]["enabled"] = False
    config["reactive_desk"]["reactive_desk_enabled"] = False
    config["camera"]["image_hw"] = 16
    config["dataset"].update(root=str(tmp_path), date="mock", compact_on_stop=False)
    monkeypatch.setattr(pipeline, "make_cameras", lambda *args: FakeCameras(config))
    monkeypatch.setattr(pipeline, "make_vr", lambda *args: FakeVR())
    stats = pipeline.run_collection(config, max_steps=60)
    assert stats["ticks"] == 60
    info = read_dataset(tmp_path / "ur5_lerobot_mock")
    assert info["total_episodes"] == 1 and info["total_frames"] > 0
    features = {item["key"]: item["shape"] for item in info["features"]}
    assert features["observation.joint_position"] == features["action.joint_position"] == [6]
    assert driver.poll() is None, "Runner shutdown stopped the independently launched driver"
