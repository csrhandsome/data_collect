"""Measure recording sessions and saved-episode deletion on disposable v3 data.

Run the same command before/after a change, reusing --work-dir so deletion reads
the exact same seed. Seed creation and copying each deletion input are untimed.
"""

import argparse
import json
import platform
import shutil
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from datasets import disable_progress_bars

from control._panda.fake import FakeBackend
from control.collection.dataset import open_dataset
from control.collection.deletion import EpisodeDeleter
from control.collection.devices import CameraPair
from control.collection.recording import EpisodeRecorder
from data_analysis.dataset_io import episode_rows


def record_session(root, episodes, frames, image_hw):
    config = {
        "dataset": {
            "repo_id": "local/benchmark",
            "date": "session",
            "root": str(root),
            "instruction": "Benchmark recording",
            "action_space": "ee",
        },
        "camera": {"image_hw": image_hw, "fps": 30},
        "control": {"frequency_hz": 100},
    }
    images = np.random.default_rng(42).integers(
        0, 256, size=(frames, image_hw, image_hw, 3), dtype=np.uint8
    )
    backend = FakeBackend()
    finish_times = []
    started = time.perf_counter()
    dataset, dataset_root = open_dataset(config)
    recorder = EpisodeRecorder(dataset, dataset_root, config)
    try:
        for _ in range(episodes):
            recorder.start()
            for image in images:
                state = backend.snapshot()
                ns = state.sampled_monotonic_ns
                recorder.frame(state, CameraPair(image, image, ns, ns, ns / 1e6, ns / 1e6))
            saving = time.perf_counter()
            recorder.finish(backend.snapshot(), success=True)
            finish_times.append(time.perf_counter() - saving)
    finally:
        closing = time.perf_counter()
        recorder.close()
    elapsed = time.perf_counter() - started
    close_seconds = time.perf_counter() - closing
    rows = episode_rows(dataset_root)
    assert len(rows) == episodes and sum(row["length"] for row in rows) == episodes * frames
    return {
        "session_seconds": elapsed,
        "finish_mean_seconds": statistics.mean(finish_times),
        "close_seconds": close_seconds,
        "data_shards": len(list((dataset_root / "data").rglob("*.parquet"))),
    }, dataset_root


def delete_episode(seed, root, index):
    shutil.copytree(seed, root)
    copied_bytes = 0
    copyfile = shutil.copyfile

    def counted_copy(source, target, *args, **kwargs):
        nonlocal copied_bytes
        copied_bytes += Path(source).stat().st_size
        return copyfile(source, target, *args, **kwargs)

    with patch.object(shutil, "copyfile", counted_copy):
        started = time.perf_counter()
        result = EpisodeDeleter(root).delete_episode(index)
        elapsed = time.perf_counter() - started
    assert len(episode_rows(root)) == result.total_episodes_after
    return {"seconds": elapsed, "copied_bytes": copied_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--image-hw", type=int, default=224)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--unique-seed-images", action="store_true", help="Avoid Parquet image deduplication in the deletion seed")
    args = parser.parse_args()
    if args.episodes < 2 or min(args.frames, args.image_hw, args.repeats) < 1:
        parser.error("episodes >= 2; frames, image-hw and repeats must be positive")
    disable_progress_bars()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    seed = args.work_dir / "seed"
    seed_config = {
        "episodes": args.episodes, "frames": args.frames, "image_hw": args.image_hw,
        "unique_seed_images": args.unique_seed_images,
    }
    if not seed.exists():
        # Seed uses the upstream session lifecycle for identical shared shards
        # regardless of which project recording implementation is benchmarked.
        from lerobot.datasets import LeRobotDataset

        from control.collection.dataset import features

        schema = features(args.image_hw, "ee", False)
        dataset = LeRobotDataset.create(
            "local/benchmark_seed", 30, schema, root=seed, metadata_buffer_size=1
        )
        rng = np.random.default_rng(17)
        image = rng.integers(
            0, 256, size=(args.image_hw, args.image_hw, 3), dtype=np.uint8
        )
        try:
            for index in range(args.episodes):
                for _ in range(args.frames):
                    if args.unique_seed_images:
                        image = rng.integers(0, 256, size=image.shape, dtype=np.uint8)
                    frame = {
                        key: image if ft["dtype"] == "image" else np.full(ft["shape"], index, dtype=np.float32)
                        for key, ft in schema.items()
                    }
                    dataset.add_frame({**frame, "task": f"Task {index}"})
                dataset.save_episode()
                (seed / f"episode_{index:06d}.actions.jsonl").write_text(
                    json.dumps({"marker": index}) + "\n"
                )
                (seed / f"episode_{index:06d}.sync.json").write_text(
                    json.dumps({"episode_index": index, "action_trace": f"episode_{index:06d}.actions.jsonl"})
                )
        finally:
            dataset.finalize()
        (args.work_dir / "seed-config.json").write_text(json.dumps(seed_config))
    else:
        stored = json.loads((args.work_dir / "seed-config.json").read_text())
        stored.setdefault("unique_seed_images", False)
        if stored != seed_config:
            parser.error("work-dir contains a seed for different parameters; select a new directory")
    seed_bytes = sum(path.stat().st_size for path in seed.rglob("*") if path.is_file())
    recordings, deletions = [], []
    # Exclude one complete warmup for each operation (imports/codecs/page cache).
    for repeat in range(args.repeats + 1):
        run = args.work_dir / f"{args.label}-{repeat}"
        run.mkdir()
        try:
            recording, _ = record_session(run / "record", args.episodes, args.frames, args.image_hw)
            deletion = delete_episode(seed, run / "delete", args.episodes // 2)
            if repeat:
                recordings.append(recording)
                deletions.append(deletion)
            print(json.dumps({"repeat": repeat, "warmup": repeat == 0, "recording": recording, "deletion": deletion}), flush=True)
        finally:
            shutil.rmtree(run)
    result = {
        "label": args.label,
        "python": platform.python_version(),
        "parameters": {**seed_config, "repeats": args.repeats, "warmups": 1},
        "seed_bytes": seed_bytes,
        "recording": recordings,
        "deletion": deletions,
        "median": {
            "recording": {key: statistics.median(row[key] for row in recordings) for key in recordings[0]},
            "deletion": {key: statistics.median(row[key] for row in deletions) for key in deletions[0]},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["median"], indent=2))


if __name__ == "__main__":
    main()
