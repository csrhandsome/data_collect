"""Control latency, failure propagation, and real MP4/Parquet integrity."""

import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import av
import numpy as np
import pyarrow.parquet as pq
import pytest
from lerobot.datasets import dataset_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from control.collection.async_recording import AsyncEpisodeRecorder
from control.collection.dataset import open_dataset
from control.collection.devices import FakeVR
from control.collection.recording import EpisodeRecorder
from control.collection.video_writer import LosslessStreamingEncoder
from control.collection.vr import VRResampler
from control.direct.fake import FakeBackend
from control.types import CameraPair
from data_analysis.compact_v3 import compact_dataset
from data_analysis.dataset_io import episode_paths
from replay.scripts.read_dataset import read_dataset
from replay.scripts.resolve_video import resolve_video


def config_for(tmp_path):
    return {
        'dataset': {'repo_id': 'local/async', 'date': 'test', 'root': str(tmp_path),
                    'instruction': 'Move', 'action_space': 'ee'},
        'camera': {'image_hw': 32, 'fps': 30}, 'control': {'frequency_hz': 100},
    }


def add_frames(recorder, base, value, count=4):
    for index in range(count):
        ns = base.sampled_monotonic_ns + index + 1
        state = replace(base, sampled_monotonic_ns=ns)
        sample = VRResampler().sample(FakeVR().latest, ns)
        image = np.full((32, 32, 3), value, dtype=np.uint8)
        recorder.trace(state, sample, state.ee_position, state.ee_quaternion_xyzw)
        recorder.frame(state, CameraPair(image, image, ns, ns, 0, 0))


def test_slow_save_allows_control_and_next_episode_then_closes_and_compacts(tmp_path):
    config = config_for(tmp_path)
    dataset, root = open_dataset(config)
    entered, proceed = threading.Event(), threading.Event()

    class SlowRecorder(EpisodeRecorder):
        def finish(self, *args, **kwargs):
            if self.index == 0:
                entered.set()
                assert proceed.wait(5)
            return super().finish(*args, **kwargs)

    recorder = AsyncEpisodeRecorder(SlowRecorder(dataset, root, config))
    base = FakeBackend().snapshot()
    try:
        recorder.start()
        add_frames(recorder, base, 40)
        begin = time.perf_counter()
        recorder.finish(base, publish=True)
        assert time.perf_counter() - begin < 0.05
        assert entered.wait(5)
        assert not recorder.active
        # The first disk save is still blocked, but capture of the next episode
        # must complete without waiting for it. The queue preserves ordering.
        begin = time.perf_counter()
        recorder.start()
        add_frames(recorder, base, 180)
        recorder.finish(base, publish=True)
        assert time.perf_counter() - begin < 0.1
        ticks = 0
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            recorder.check()
            ticks += 1
            time.sleep(0.01)
        assert ticks >= 10 and not proceed.is_set()
    finally:
        proceed.set()
        recorder.close()
    assert read_dataset(root)['total_frames'] == 8
    assert compact_dataset(root)
    for index, expected in enumerate((40, 180)):
        source = resolve_video(root, index, 'observation.exterior_image')
        with av.open(str(source['path'])) as container:
            frames = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)
                      if source['start_time_s'] - 1e-6 <= float(f.pts * f.time_base)
                      < source['end_time_s'] - 1e-6]
        assert len(frames) == 4
        assert np.mean(frames) == pytest.approx(expected, abs=3)
        sync = json.loads((root / f'episode_{index:06d}.sync.json').read_text())
        assert sync['action_records'] == sync['video_frames'] == 4
        assert sync['frame_records'][-1]['terminal_action']
    assert not list(root.rglob('*.png'))


def test_full_queue_fails_immediately_and_close_drains_accepted_work(tmp_path):
    entered, proceed = threading.Event(), threading.Event()
    calls = []

    def start(**kwargs):
        entered.set()
        assert proceed.wait(5)

    raw = SimpleNamespace(root=tmp_path, start=start,
                          enter_critical_phase=lambda ns: calls.append(ns),
                          close=lambda: calls.append('closed'))
    recorder = AsyncEpisodeRecorder(raw, max_pending=2)
    try:
        recorder.start()
        assert entered.wait(5)
        recorder.enter_critical_phase(1)
        recorder.enter_critical_phase(2)
        begin = time.perf_counter()
        with pytest.raises(RuntimeError, match='backlog is full'):
            recorder.enter_critical_phase(3)
        assert time.perf_counter() - begin < 0.05
    finally:
        proceed.set()
        recorder.close()
    assert calls == [1, 2, 'closed']


def test_worker_error_is_visible_while_idle_and_at_shutdown(tmp_path):
    failed = threading.Event()
    closed = []

    def start(**kwargs):
        failed.set()
        raise OSError('disk full')

    raw = SimpleNamespace(root=tmp_path, start=start, close=lambda: closed.append(True))
    recorder = AsyncEpisodeRecorder(raw)
    recorder.start()
    assert failed.wait(5)
    recorder._queue.join()
    with pytest.raises(RuntimeError, match='Episode I/O failed') as exc:
        recorder.check()
    assert isinstance(exc.value.__cause__, OSError)
    with pytest.raises(RuntimeError, match='Episode I/O failed'):
        recorder.close()
    assert closed == [True]


def test_video_writer_skips_image_map_and_preserves_upstream_arrow_schema(tmp_path, monkeypatch):
    def forbid_map(*args, **kwargs):
        pytest.fail('Video datasets must not call image embedding map')

    monkeypatch.setattr(dataset_writer, 'embed_images', forbid_map)
    config = config_for(tmp_path)
    dataset, root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, root, config)
    base = FakeBackend().snapshot()
    try:
        for value in (60, 100):
            recorder.start()
            add_frames(recorder, base, value)
            recorder.finish(base, publish=True)
    finally:
        recorder.close()
    assert not (root / 'images').exists()
    table = pq.read_table(episode_paths(root)[0])
    assert table['frame_index'].to_pylist() == [0, 1, 2, 3]
    assert table['observation.joint_position'].type.list_size == 7
    assert b'huggingface' in table.schema.metadata
    # Training uses the actual LeRobot reader, including its video timestamp offsets.
    reader = LeRobotDataset('local/async_test', root=root, video_backend='pyav', return_uint8=True)
    for index, value in ((0, 60), (7, 100)):
        row = reader[index]
        assert float(row['observation.exterior_image'].float().mean()) == pytest.approx(value, abs=3)


def test_dead_encoder_with_full_queue_fails_instead_of_hanging():
    import queue

    encoder = LosslessStreamingEncoder(30)
    encoder._episode_active = True
    frames = queue.Queue(maxsize=1)
    frames.put('unconsumed')
    encoder._frame_queues['camera'] = frames
    encoder._threads['camera'] = SimpleNamespace(is_alive=lambda: False)
    begin = time.perf_counter()
    with pytest.raises(RuntimeError, match='Video encoder failed'):
        encoder.feed_frame('camera', np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match='failed before flush'):
        encoder.finish_episode()
    assert time.perf_counter() - begin < 0.05


def test_pipeline_keeps_sending_targets_during_slow_disk_save(tmp_path, monkeypatch):
    from control.collection import pipeline
    from control.config import load_config

    config = load_config()
    config['dataset'].update(root=str(tmp_path), date='slow_disk')
    config['camera']['image_hw'] = 32
    config['audio'].update(enabled=False, vad_enabled=False)
    entered, proceed = threading.Event(), threading.Event()
    sending = []
    original_finish = EpisodeRecorder.finish
    backend = FakeBackend()
    target = backend.target

    def finish(recorder, *args, **kwargs):
        if recorder.active and recorder.index == 0:
            entered.set()
            assert proceed.wait(5), 'Control loop stalled behind episode save'
        return original_finish(recorder, *args, **kwargs)

    def send(*args):
        if entered.is_set() and not proceed.is_set():
            sending.append(time.monotonic())
            if len(sending) >= 12:
                proceed.set()
        return target(*args)

    class SaveVR(FakeVR):
        reads = 0

        @property
        def latest(self):
            sample = super().latest
            self.reads += 1
            sample.y_pressed = self.reads == 20
            sample.arm_enabled = self.reads != 21
            return sample

    monkeypatch.setattr(EpisodeRecorder, 'finish', finish)
    monkeypatch.setattr(backend, 'target', send)
    monkeypatch.setattr('control.direct.fake.FakeBackend', lambda: backend)
    monkeypatch.setattr(pipeline, 'make_vr', lambda *_: SaveVR())
    stats = pipeline.run_collection(config, dry_run=True, max_steps=60)
    assert stats['ticks'] == 60
    assert len(sending) == 12
    assert max(np.diff(sending)) < 0.08
    assert len(list(tmp_path.rglob('*.sync.json'))) == 2


def test_truncated_encoder_output_is_never_published(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    dataset, root = open_dataset(config)
    encoder = dataset.writer._streaming_encoder
    feed = encoder.feed_frame
    skipped = set()

    def drop_one(key, image):
        if key not in skipped:
            skipped.add(key)
            return
        feed(key, image)

    monkeypatch.setattr(encoder, 'feed_frame', drop_one)
    recorder = EpisodeRecorder(dataset, root, config)
    base = FakeBackend().snapshot()
    try:
        recorder.start()
        add_frames(recorder, base, 100)
        with pytest.raises(RuntimeError, match='Video frame mismatch'):
            recorder.finish(base, publish=True)
        assert not list(root.glob('*.sync.json'))
        assert dataset.meta.total_episodes == 0
    finally:
        recorder.close()


def test_numeric_writer_rollover_retains_global_indices_and_task_schema(tmp_path):
    config = config_for(tmp_path)
    config['observation'] = {'exterior_image': False, 'wrist_image_left': False}
    dataset, root = open_dataset(config)
    dataset.meta.info.data_files_size_in_mb = 0.000001
    recorder = EpisodeRecorder(dataset, root, config)
    base = FakeBackend().snapshot()
    try:
        for index in range(3):
            config['dataset']['instruction'] = f'Task {index}'
            recorder.start()
            add_frames(recorder, base, 40)
            recorder.finish(base)
    finally:
        recorder.close()
    paths = episode_paths(root)
    assert len(set(paths.values())) == 3
    for index, path in paths.items():
        rows = pq.read_table(path)
        assert rows['index'].to_pylist() == list(range(index * 4, index * 4 + 4))
        assert rows['task_index'].to_pylist() == [index] * 4
        assert rows['episode_index'].to_pylist() == [index] * 4
    reader = LeRobotDataset('local/async_test', root=root)
    assert reader[11]['task'] == 'Task 2'
