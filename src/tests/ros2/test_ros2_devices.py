"""Image decoding and real DDS sensor subscriptions, including episode audio."""

import json
import os
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from control.collection.devices import make_cameras, make_microphone
from control.config import load_config, validate
from control.ros2.camera_connector import image_rgb


@pytest.mark.parametrize("encoding,count", [("rgb8", 3), ("bgr8", 3), ("rgba8", 4),
                                            ("bgra8", 4), ("mono8", 1)])
def test_image_decodes_channels_and_padded_rows(encoding, count):
    row = [10, 20, 30, 255][:count] + [99, 99]
    msg = SimpleNamespace(encoding=encoding, height=2, width=1, step=count + 2,
                          data=bytes(row * 2))
    expected = [10] * 3 if count == 1 else ([30, 20, 10] if encoding.startswith("bgr") else [10, 20, 30])
    np.testing.assert_array_equal(image_rgb(msg), np.array([[expected], [expected]], dtype=np.uint8))


@pytest.mark.parametrize("override", [{"encoding": "16UC1"}, {"height": 0},
                                      {"step": 2}, {"data": b""}])
def test_image_rejects_malformed_payloads(override):
    fields = dict(encoding="rgb8", height=1, width=1, step=3, data=b"\x01\x02\x03")
    fields.update(override)
    with pytest.raises(ValueError):
        image_rgb(SimpleNamespace(**fields))


@pytest.mark.parametrize("section,key,value", [
    ("camera", "backend", "unknown"), ("audio", "backend", "unknown"),
    ("camera", "max_age_s", 0),
])
def test_sensor_config_rejects_invalid_settings(section, key, value):
    config = load_config()
    config["camera"]["backend"] = "ros2"
    config[section][key] = value
    with pytest.raises(ValueError):
        validate(config)


ROS_TEST = pytest.mark.skipif(
    os.environ.get("DATA_COLLECT_TEST_ROS2") != "1", reason="Requires ROS 2 and DDS sockets",
)


@pytest.fixture
def sensor_peer(tmp_path):
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import Int16MultiArray

    context = Context()
    rclpy.init(args=[], context=context)
    namespace = "/sensors_" + uuid4().hex[:8]
    node = rclpy.create_node("peer", namespace=namespace, context=context)
    publishers = {
        name: node.create_publisher(Image, name, qos_profile_sensor_data)
        for name in ("external", "wrist")
    }
    audio = node.create_publisher(Int16MultiArray, "audio", qos_profile_sensor_data)
    peer = SimpleNamespace(images=True, audio=True, source_age_s=0)

    def publish():
        if peer.images:
            for i, publisher in enumerate(publishers.values()):
                msg = Image(height=16, width=16, encoding="rgb8", step=48)
                ns = node.get_clock().now().nanoseconds - round(peer.source_age_s * 1e9)
                msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 1_000_000_000)
                msg.data = np.full((16, 16, 3), 30 + i, dtype=np.uint8).tobytes()
                publisher.publish(msg)
        if peer.audio:
            audio.publish(Int16MultiArray(data=[123] * 320))

    node.create_timer(0.02, publish)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    parent = Path(__file__).resolve().parents[3] / "config/train/panda.yaml"
    path = tmp_path / "sensors.yaml"
    path.write_text(f"extends: {parent}\nrobot:\n  backend: ros2\n")
    peer.config = load_config(path)
    peer.config["camera"].update(image_hw=16, crop_scale=1.0, max_age_s=0.3, startup_timeout_s=2)
    peer.config["camera"]["ros2"].update(
        external_image_topic=namespace + "/external", wrist_image_topic=namespace + "/wrist",
    )
    peer.config["audio"].update(enabled=True, sample_rate=16000, channels=1)
    peer.config["audio"]["ros2"].update(topic=namespace + "/audio", startup_timeout_s=2)
    peer.config["ros2"]["enabled"] = False
    try:
        yield peer
    finally:
        executor.shutdown(timeout_sec=2)
        thread.join(timeout=2)
        node.destroy_node()
        context.try_shutdown()
        context.destroy()


@ROS_TEST
def test_dds_camera_audio_and_recording(sensor_peer, tmp_path):
    config = sensor_peer.config
    cameras = make_cameras(config)
    microphone = None
    try:
        pair = cameras.get_frames()
        assert pair.front.shape == pair.wrist.shape == (16, 16, 3)
        assert pair.front[0, 0, 0] == 30 and pair.wrist[0, 0, 0] == 31
        microphone = make_microphone(config)
        path = tmp_path / "audio.wav"
        microphone.default_output_path = path
        assert microphone.start_recording()
        microphone._source.wait(lambda: microphone._samples >= 640, 2)
        assert microphone.stop_recording() == path
        with wave.open(str(path), "rb") as wav:
            assert wav.getframerate() == 16000 and wav.getnchannels() == 1
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            assert samples.size >= 640 and np.all(samples == 123)
        metadata = json.loads(path.with_suffix(".audio.json").read_text())
        assert metadata["num_samples"] == samples.size
        assert metadata["timestamp_source"] == "host_receipt"
        assert metadata["chunk_timestamps_ns"] == sorted(metadata["chunk_timestamps_ns"])
        microphone.default_output_path = None
        assert microphone.start_recording()
        microphone._source.wait(lambda: microphone._samples >= 320, 2)
        assert microphone.stop_recording() is None
        assert microphone.last_metadata_path is None
        sensor_peer.images = False
        time.sleep(0.35)
        with pytest.raises(TimeoutError, match="stale"):
            cameras.get_frames()
        sensor_peer.audio = False
        time.sleep(0.55)
        with pytest.raises(TimeoutError, match="stale"):
            microphone.start_recording()
    finally:
        if microphone is not None:
            microphone.stop()
        cameras.close()


@ROS_TEST
def test_collection_records_subscribed_images_and_wav(sensor_peer, tmp_path, monkeypatch):
    from control.collection import pipeline
    from control.collection.devices import FakeVR
    from control.direct.fake import FakeBackend
    from control.runners import create_runner

    config = sensor_peer.config
    config["dataset"].update(root=str(tmp_path), date="ros_sensors", enable_logging=True)
    config["gripper"]["type"] = "none"
    config["reactive_desk"]["reactive_desk_enabled"] = False
    monkeypatch.setattr(pipeline, "make_vr", lambda config, dry_run: FakeVR())
    monkeypatch.setattr(pipeline, "create_runner", lambda **kw: create_runner(config=kw["config"], backend=FakeBackend()))
    result = pipeline.run_collection(config, max_steps=40)
    assert result["ticks"] == 40
    sync_path, = tmp_path.rglob("episode_*.sync.json")
    sync = json.loads(sync_path.read_text())
    assert sync["frame_records"]
    wav_path, = tmp_path.rglob("episode_*.wav")
    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getnframes() > 0
    assert json.loads(wav_path.with_suffix(".audio.json").read_text())["source"] == "ros2"


@ROS_TEST
def test_stale_camera_source_times_out_and_cleans_up(sensor_peer):
    sensor_peer.source_age_s = 10
    sensor_peer.config["camera"]["startup_timeout_s"] = 0.15
    with pytest.raises(TimeoutError, match="No valid camera"):
        make_cameras(sensor_peer.config)


@ROS_TEST
def test_missing_microphone_times_out(sensor_peer):
    sensor_peer.audio = False
    sensor_peer.config["audio"]["ros2"]["startup_timeout_s"] = 0.15
    with pytest.raises(TimeoutError, match="No valid microphone"):
        make_microphone(sensor_peer.config)


@ROS_TEST
def test_audio_loss_rejects_save_and_discards_staged_wav(sensor_peer, tmp_path):
    microphone = make_microphone(sensor_peer.config)
    try:
        path = tmp_path / "missing.wav"
        microphone.default_output_path = path
        assert microphone.start_recording()
        microphone._source.wait(lambda: microphone._samples >= 320, 2)
        sensor_peer.audio = False
        time.sleep(0.55)
        with pytest.raises(TimeoutError, match="No fresh"):
            microphone.stop_recording()
        assert not path.exists()
        assert microphone._temp is None and not microphone.is_recording
    finally:
        microphone.stop()
