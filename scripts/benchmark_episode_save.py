"""Hardware-free save benchmark; all datasets live in a temporary directory.

PYTHONPATH=src .venv/bin/python scripts/benchmark_episode_save.py --mode baseline
PYTHONPATH=src .venv/bin/python scripts/benchmark_episode_save.py --mode optimized
Defaults to 30 fps; --no-pace stress-tests the bounded queues instead.
"""

import argparse
import json
import logging
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from control.collection import recording
from control.collection.async_recording import AsyncEpisodeRecorder
from control.collection.dataset import open_dataset
from control.collection.schema import features
from control.direct.fake import FakeBackend
from control.types import CameraPair
from data_analysis.compact_v3 import compact_dataset
from replay.scripts.resolve_video import resolve_video


def benchmark(mode, lengths, pace):
    with tempfile.TemporaryDirectory(prefix="episode-save-benchmark-") as folder:
        config = {
            "dataset": {"repo_id": "local/benchmark", "date": "save", "root": folder,
                        "instruction": "Move", "action_space": "ee"},
            "camera": {"image_hw": 224, "fps": 30}, "control": {"frequency_hz": 100},
        }
        root = Path(folder) / "benchmark_save"
        encoder = RGBEncoderConfig(vcodec="h264", crf=18, g=2, preset="fast")

        def baseline_open(_):
            kwargs = dict(repo_id="local/benchmark_save", root=root, rgb_encoder=encoder,
                          encoder_threads=2, image_writer_threads=4, image_writer_processes=0)
            if root.exists():
                return LeRobotDataset.resume(**kwargs), root
            return LeRobotDataset.create(
                **kwargs, fps=30, robot_type="panda", features=features(224, "ee", False),
                use_videos=True, metadata_buffer_size=1,
            ), root

        saves, opens, calls = [], [], []

        class TimedRecorder(recording.EpisodeRecorder):
            def start(self, **kwargs):
                begin = time.perf_counter()
                super().start(**kwargs)
                opens.append(time.perf_counter() - begin)

            def finish(self, *args, **kwargs):
                begin = time.perf_counter()
                super().finish(*args, **kwargs)
                saves.append(time.perf_counter() - begin)

        factory = baseline_open if mode == "baseline" else open_dataset
        original = recording.open_dataset
        recording.open_dataset = factory
        dataset, root = factory(config)
        raw = TimedRecorder(dataset, root, config)
        recorder = raw if mode == "baseline" else AsyncEpisodeRecorder(raw)
        base = FakeBackend().snapshot()
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        start = time.perf_counter()
        try:
            for episode, length in enumerate(lengths):
                recorder.start()
                begin = time.perf_counter()
                for frame in range(length):
                    if pace:
                        time.sleep(max(0, begin + frame / 30 - time.perf_counter()))
                    ns = base.sampled_monotonic_ns + (sum(lengths[:episode]) + frame + 1) * 33_333_333
                    state = replace(base, sampled_monotonic_ns=ns)
                    pixels = np.roll(image, frame % 224, axis=1)
                    recorder.frame(state, CameraPair(pixels, pixels, ns, ns, 0, 0))
                begin = time.perf_counter()
                recorder.finish(state, publish=True)
                calls.append(time.perf_counter() - begin)
            close_start = time.perf_counter()
            recorder.close()
            close_s = time.perf_counter() - close_start
            elapsed_s = time.perf_counter() - start
            compact_start = time.perf_counter()
            compact_dataset(root)
            compact_s = time.perf_counter() - compact_start
            decoded = []
            for index, expected in enumerate(lengths):
                for key in dataset.meta.video_keys:
                    source = resolve_video(root, index, key)
                    with av.open(str(source["path"])) as container:
                        count = sum(source['start_time_s'] - 1e-6 <= float(f.pts * f.time_base)
                                    < source['end_time_s'] - 1e-6 for f in container.decode(video=0))
                    assert count == expected, (index, key, count, expected)
                    decoded.append(count)
            return {"mode": mode, "frames": lengths, "paced": pace, "open_s": opens,
                    "save_worker_s": saves, "save_control_call_s": calls,
                    "close_wait_s": close_s, "record_and_save_s": elapsed_s,
                    "compact_s": compact_s, "decoded_counts": decoded,
                    "remaining_png": len(list(root.rglob("*.png")))}
        finally:
            recording.open_dataset = original
            recorder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "optimized"], required=True)
    parser.add_argument("--frames", nargs="+", type=int, default=[1135, 114])
    parser.add_argument("--pace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-save-s", type=float, help="Fail if any background save exceeds this budget")
    parser.add_argument("--max-control-ms", type=float, help="Fail if submitting a save exceeds this budget")
    args = parser.parse_args()
    if any(n <= 0 for n in args.frames):
        parser.error("frame counts must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    result = benchmark(args.mode, args.frames, args.pace)
    print(json.dumps(result, indent=2))
    if args.max_save_s is not None and max(result["save_worker_s"]) > args.max_save_s:
        raise SystemExit("Background save exceeded --max-save-s")
    if (args.max_control_ms is not None
            and max(result["save_control_call_s"]) * 1000 > args.max_control_ms):
        raise SystemExit("Control call exceeded --max-control-ms")


if __name__ == "__main__":
    main()
