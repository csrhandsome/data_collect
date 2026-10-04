"""Cache embedded camera images as browser-playable H.264, without changing source data.

uv run python -m replay.scripts.prepare_video DATASET --episode 0
"""

import argparse
import fcntl
import hashlib
import io
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from ._common import _episode_data_path, _episode_table, _get_episode, _load_metadata, _safe_path
from .read_sidecars import confined, origin_ns, read_sync

_CACHE_VERSION = 1


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def _stamp(root, path):
    path = _safe_path(root, Path(path).relative_to(root))
    stat = path.stat()
    return [str(path.relative_to(root)), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size]


def _source_stamp(root, info, episode, feature):
    paths = [_episode_data_path(root, info, episode)]
    for folder in ["audio", ""]:
        path = confined(
            root, f"{folder}/episode_{episode['episode_index']:06d}.sync.json".lstrip("/")
        )
        if path.is_file():
            paths.append(path)
    return _digest([_CACHE_VERSION, info, episode, feature, [_stamp(root, p) for p in paths]])


def _timestamps(root, info, episode):
    sync = read_sync(root, episode["episode_index"])
    frames = sync.get("frame_records", [])
    if frames:
        if len(frames) != episode["length"]:
            raise ValueError("Sync image frame count disagrees with episode")
        timestamps = np.array(
            [(r["host_frame_monotonic_ns"] - origin_ns(sync)) / 1e9 for r in frames]
        )
    else:
        timestamps = np.arange(episode["length"]) / info["fps"]
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Image timestamps must be finite and increasing")
    return timestamps


def _rgb(root, value, dependencies):
    if isinstance(value, dict):
        data = value.get("bytes")
        if data is None and value.get("path"):
            path = _safe_path(root, value["path"])
            dependencies[str(path.relative_to(root))] = _stamp(root, path)
            data = path.read_bytes()
    else:
        data = value
    if not isinstance(data, bytes):
        raise ValueError("Embedded image bytes are missing")
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def _encode(root, table, feature, timestamps, fps, output, dependencies):
    """Resample to camera FPS using host timestamps; hold frames across capture gaps."""
    first = _rgb(root, table[feature][0].as_py(), dependencies)
    width, height = first.size
    duration = float(timestamps[-1]) + 1 / fps
    count = max(1, math.ceil(duration * fps - 1e-7))
    target = np.arange(count) / fps
    right = np.clip(np.searchsorted(timestamps, target), 0, len(timestamps) - 1)
    left = np.maximum(0, right - 1)
    indices = np.where(
        abs(target - timestamps[left]) <= abs(timestamps[right] - target), left, right
    )
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-g",
        str(max(1, round(fps))),
        "-threads",
        "2",
        "-movflags",
        "+faststart",
        str(output),
    ]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
        )
        try:
            previous, raw = -1, None
            for index in indices:
                if index != previous:
                    image = (
                        first
                        if index == 0
                        else _rgb(root, table[feature][int(index)].as_py(), dependencies)
                    )
                    if image.size != (width, height):
                        raise ValueError("Camera frame dimensions changed inside an episode")
                    raw, previous = image.tobytes(), index
                process.stdin.write(raw)
            process.stdin.close()
            if process.wait(timeout=120) != 0:
                raise RuntimeError("Camera MP4 encoding failed")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            if not process.stdin.closed:
                process.stdin.close()
    return duration


def prepare_video(root, episode_index, feature):
    """Single-flight, atomically published cache, invalidated by source file changes."""
    root = Path(root).resolve()
    info, episodes = _load_metadata(root)
    episode = _get_episode(episodes, episode_index)
    if feature not in info["features"]:
        raise KeyError(feature)
    if info["features"][feature]["dtype"] != "image":
        raise ValueError("Feature is not an embedded camera stream")
    folder = _safe_path(
        root, f".replay-cache/video/{episode_index:06d}/{_digest(feature)}", require_file=False
    )
    folder.mkdir(parents=True, exist_ok=True)
    lock = _safe_path(root, folder.relative_to(root) / "encode.lock", require_file=False)
    manifest = _safe_path(root, folder.relative_to(root) / "current.json", require_file=False)
    with lock.open("a+b") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        # Re-read after waiting: collection/deletion may have changed the episode.
        info, episodes = _load_metadata(root)
        episode = _get_episode(episodes, episode_index)
        source = _source_stamp(root, info, episode, feature)
        cached = None
        if manifest.is_file():
            try:
                cached = json.loads(manifest.read_text())
            except (ValueError, OSError):
                pass
        if isinstance(cached, dict) and cached.get("source") == source:
            fresh = all(
                _stamp(root, root / p) == stamp for p, stamp in cached["dependencies"].items()
            )
            path = _safe_path(root, cached["path"], require_file=False)
            if fresh and path.is_file() and path.stat().st_size:
                return {**cached["video"], "path": path}

        table = _episode_table(root, info, episode, feature)
        dependencies = {}
        timestamps = _timestamps(root, info, episode)
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", prefix="encoding-", dir=folder, delete=False
        ) as staged_video:
            temporary = Path(staged_video.name)
        try:
            duration = _encode(
                root, table, feature, timestamps, info["fps"], temporary, dependencies
            )
            latest_info, latest_episodes = _load_metadata(root)
            if (
                _source_stamp(
                    root, latest_info, _get_episode(latest_episodes, episode_index), feature
                )
                != source
            ):
                raise ValueError("Dataset changed during video preparation; retry")
            if any(_stamp(root, root / p) != stamp for p, stamp in dependencies.items()):
                raise ValueError("Camera images changed during video preparation; retry")
            version = _digest([source, dependencies])
            path = _safe_path(root, folder.relative_to(root) / f"{version}.mp4", require_file=False)
            temporary.replace(path)
            video = {
                "feature": feature,
                "start_time_s": 0.0,
                "end_time_s": duration,
                "duration_s": duration,
                "fps": info["fps"],
                "version": version,
            }
            payload = {
                "source": source,
                "dependencies": dependencies,
                "path": str(path.relative_to(root)),
                "video": video,
            }
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", dir=folder, delete=False
            ) as staged:
                json.dump(payload, staged)
            Path(staged.name).replace(manifest)
            for old in folder.glob("*.mp4"):
                if old != path:
                    old.unlink()
            return {**video, "path": path}
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--episode", type=int, action="append", help="Repeat to select episodes; default: all"
    )
    parser.add_argument(
        "--feature", action="append", help="Repeat to select cameras; default: all images"
    )
    args = parser.parse_args()
    info, episodes = _load_metadata(args.dataset)
    features = args.feature or [k for k, v in info["features"].items() if v["dtype"] == "image"]
    for episode in args.episode if args.episode is not None else episodes:
        for feature in features:
            start = time.perf_counter()
            video = prepare_video(args.dataset, episode, feature)
            print(
                json.dumps(
                    {
                        **video,
                        "path": str(video["path"]),
                        "prepare_s": round(time.perf_counter() - start, 4),
                        "size_bytes": video["path"].stat().st_size,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
