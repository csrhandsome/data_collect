"""FTP-1 ViTaMIn: AllTact Fin Ray episodes stored as zarr inside ViTaMIn.tar.

Each finger has an internal RGB camera. Those frames land on the two gripper
image slots. The GoPro on the handheld gripper lands on the wrist image.
"""

from __future__ import annotations

import ctypes
import io
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from control.util.pose import matrix_to_rpy
from data_analysis.data_clean.base import BaseAligner
from data_analysis.data_clean.lerobot_v3 import (
    GRIPPER_IMAGE_LEFT,
    GRIPPER_IMAGE_RIGHT,
    AlignedEpisode,
)

_SCENE_KEYS = (
    ("observation.exterior_image", ("camera_ego_rgb", "camera_main_rgb")),
    ("observation.wrist_image_left", ("right_wrist_camera_rgb", "left_wrist_camera_rgb")),
)


class ViTaMInAligner(BaseAligner):
    name = "vitamin"
    robot_type = "vitamin"

    def episodes(self, source: Path) -> list[str]:
        bundle = _open_bundle(source)
        found = []
        for store in bundle.stores():
            ends = _read_array(store, "meta/episode_ends")
            for index in range(len(ends)):
                found.append(f"{store.label}::{index}")
        return found

    def load(self, source: Path, episode_id: str, work: Path) -> AlignedEpisode:
        label, index_text = episode_id.rsplit("::", 1)
        store = _open_bundle(source).store(label)
        ends = _read_array(store, "meta/episode_ends").astype(np.int64)
        index = int(index_text)
        start = 0 if index == 0 else int(ends[index - 1])
        end = int(ends[index])
        if end <= start:
            raise RuntimeError(f"{episode_id} is empty")
        keys = set(store.arrays())
        work.mkdir(parents=True, exist_ok=True)
        fps = _fps(store, keys, start, end)
        videos = _write_videos(store, keys, start, end, work, fps)
        columns = _columns(store, keys, start, end)
        if not columns:
            zeros = np.zeros((end - start, 1), dtype=np.float32)
            columns = {"observation.gripper_position": zeros, "action.gripper_position": zeros}
        return AlignedEpisode(
            key=episode_id,
            task=_task(store, keys, label, start, end),
            fps=fps,
            videos=videos,
            columns=columns,
        )


class _Store:
    label: str

    def read(self, rel: str) -> bytes:
        raise NotImplementedError

    def arrays(self) -> list[str]:
        raise NotImplementedError


class _DirectoryStore(_Store):
    def __init__(self, root: Path):
        self.root = root
        self.label = root.name

    def read(self, rel: str) -> bytes:
        path = self.root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_bytes()

    def arrays(self) -> list[str]:
        names = []
        data = self.root / "data"
        if not data.is_dir():
            return names
        for path in sorted(data.glob("*/.zarray")):
            names.append(path.parent.name)
        return names


class _TarStore(_Store):
    def __init__(self, bundle: _TarBundle, label: str):
        self.bundle = bundle
        self.label = label
        self.prefix = f"{label}/" if "/" not in label else f"{label.strip('/')}/"

    def read(self, rel: str) -> bytes:
        return self.bundle.read(self.prefix + rel)

    def arrays(self) -> list[str]:
        prefix = self.prefix + "data/"
        names = []
        for name in self.bundle.names():
            if name.startswith(prefix) and name.endswith("/.zarray"):
                key = name[len(prefix) : -len("/.zarray")]
                if "/" not in key:
                    names.append(key)
        return sorted(names)


class _TarBundle:
    def __init__(self, path: Path):
        self.path = path
        self._index = _tar_index(path)

    def names(self):
        return self._index

    def stores(self) -> list[_TarStore]:
        labels = []
        seen = set()
        for name in self._index:
            if not name.endswith("/meta/episode_ends/.zarray"):
                continue
            label = name[: -len("/meta/episode_ends/.zarray")]
            if label not in seen:
                seen.add(label)
                labels.append(label)
        return [_TarStore(self, label) for label in sorted(labels)]

    def store(self, label: str) -> _TarStore:
        for store in self.stores():
            if store.label == label or store.label.endswith("/" + label) or Path(store.label).name == label:
                return store
        raise KeyError(label)

    def read(self, name: str) -> bytes:
        offset, size = self._index[name]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(size)
        if len(payload) != size:
            raise RuntimeError(f"{name} is short inside {self.path}")
        return payload


def _open_bundle(source: Path):
    source = Path(source)
    if source.is_file():
        return _TarBundle(source)
    zarrs = _directory_zarrs(source)
    if zarrs:
        return _DirectoryBundle(zarrs)
    tar = source / "ViTaMIn.tar"
    if not tar.is_file():
        found = sorted(source.glob("*.tar"))
        tar = found[0] if found else None
    if tar is None:
        raise FileNotFoundError(source / "ViTaMIn.tar")
    return _TarBundle(tar)


class _DirectoryBundle:
    def __init__(self, roots: list[Path]):
        self.roots = roots

    def stores(self) -> list[_DirectoryStore]:
        return [_DirectoryStore(root) for root in self.roots]

    def store(self, label: str) -> _DirectoryStore:
        for root in self.roots:
            if root.name == label or str(root).endswith(label):
                return _DirectoryStore(root)
        raise KeyError(label)


def _directory_zarrs(source: Path) -> list[Path]:
    found = []
    for pattern in ("*.zarr", "*/*.zarr"):
        for path in sorted(source.glob(pattern)):
            if (path / "meta" / "episode_ends" / ".zarray").is_file():
                found.append(path)
    return found


def _tar_index(path: Path) -> dict[str, tuple[int, int]]:
    stat = path.stat()
    cache = path.with_name(path.name + ".index.json")
    if cache.is_file():
        payload = json.loads(cache.read_text())
        if payload.get("size") == stat.st_size and payload.get("mtime_ns") == stat.st_mtime_ns:
            return {row[0]: (row[1], row[2]) for row in payload["members"]}
    print(f"[vitamin] indexing {path}", flush=True)
    members = []
    long_name = None
    with path.open("rb") as handle:
        while True:
            header = handle.read(512)
            if len(header) < 512 or header == b"\0" * 512:
                break
            size = int(header[124:136].split(b"\0", 1)[0].strip() or b"0", 8)
            kind = chr(header[156])
            data_at = handle.tell()
            padded = (size + 511) // 512 * 512
            if kind == "L":
                raw = handle.read(size)
                long_name = raw.split(b"\0", 1)[0].decode("utf-8", "replace")
                handle.seek(data_at + padded)
                continue
            name = header[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
            prefix = header[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
            if prefix:
                name = f"{prefix}/{name}"
            if long_name:
                name = long_name
                long_name = None
            if kind in {"0", "\0"}:
                members.append((name, data_at, size))
            handle.seek(data_at + padded)
    cache.write_text(
        json.dumps({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "members": members})
    )
    return {name: (offset, size) for name, offset, size in members}


def _read_array(store: _Store, rel: str, start: int = 0, end: int | None = None) -> np.ndarray:
    meta = json.loads(store.read(f"{rel}/.zarray"))
    if meta.get("filters"):
        raise RuntimeError(f"{rel} uses zarr filters this reader does not decode")
    shape = tuple(int(item) for item in meta["shape"])
    chunks = tuple(int(item) for item in meta["chunks"])
    dtype = np.dtype(meta["dtype"])
    stop = shape[0] if end is None else min(end, shape[0])
    start = max(0, start)
    if stop < start:
        raise RuntimeError(f"{rel} slice {start}:{stop} is outside {shape}")
    out_shape = (stop - start, *shape[1:])
    out = np.empty(out_shape, dtype=dtype)
    if out.size == 0:
        return out
    separator = meta.get("dimension_separator") or "."
    counts = [(shape[axis] + chunks[axis] - 1) // chunks[axis] for axis in range(len(shape))]
    first = start // chunks[0]
    last = (stop - 1) // chunks[0]
    for chunk0 in range(first, last + 1):
        block = _read_chunk(store, rel, meta, (chunk0,), counts, separator)
        block_start = chunk0 * chunks[0]
        take_from = max(start, block_start) - block_start
        take_to = min(stop, block_start + block.shape[0]) - block_start
        dest_from = max(start, block_start) - start
        dest_to = dest_from + (take_to - take_from)
        out[dest_from:dest_to] = block[take_from:take_to]
    return out


def _read_chunk(store: _Store, rel: str, meta: dict, origin: tuple[int, ...], counts: list[int], separator: str):
    shape = tuple(int(item) for item in meta["shape"])
    chunks = tuple(int(item) for item in meta["chunks"])
    dtype = np.dtype(meta["dtype"])
    valid0 = min(chunks[0], shape[0] - origin[0] * chunks[0])
    assembled = np.empty((valid0, *shape[1:]), dtype=dtype)
    spatial = [range(count) for count in counts[1:]] or [range(1)]
    for extra in itertools.product(*spatial):
        coord = (origin[0], *extra) if len(shape) > 1 else (origin[0],)
        if len(coord) < len(shape):
            coord = tuple(coord) + (0,) * (len(shape) - len(coord))
        name = separator.join(str(item) for item in coord[: len(shape)])
        raw = _decode_chunk(store.read(f"{rel}/{name}"), meta)
        valid = [min(chunks[axis], shape[axis] - coord[axis] * chunks[axis]) for axis in range(len(shape))]
        values = np.frombuffer(raw, dtype=dtype)
        if values.size == int(np.prod(chunks)):
            values = values.reshape(chunks)[tuple(slice(0, length) for length in valid)]
        else:
            values = values.reshape(valid)
        dest = [slice(0, valid[0])]
        for axis in range(1, len(shape)):
            begin = coord[axis] * chunks[axis]
            dest.append(slice(begin, begin + valid[axis]))
        assembled[tuple(dest)] = values
    return assembled


def _decode_chunk(payload: bytes, meta: dict) -> bytes:
    compressor = meta.get("compressor")
    if not compressor:
        return payload
    if not isinstance(compressor, dict) or compressor.get("id") != "blosc":
        raise RuntimeError(f"unsupported zarr compressor: {compressor}")
    return _blosc_decompress(payload)


_BLOSC = None


def _blosc_library():
    global _BLOSC
    if _BLOSC is not None:
        return _BLOSC
    candidates = ["libblosc.so.1", "libblosc.so", str(Path.home() / ".local" / "lib" / "libblosc.so.1")]
    last = None
    for name in candidates:
        try:
            library = ctypes.CDLL(name)
        except OSError as exc:
            last = exc
            continue
        library.blosc_decompress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        library.blosc_decompress.restype = ctypes.c_int
        _BLOSC = library
        return library
    raise RuntimeError(f"找不到 libblosc.so.1（可放到 ~/.local/lib）: {last}")


def _blosc_decompress(payload: bytes) -> bytes:
    if len(payload) < 16:
        raise RuntimeError("blosc chunk is shorter than its header")
    nbytes = int.from_bytes(payload[4:8], "little")
    source = ctypes.create_string_buffer(payload)
    dest = ctypes.create_string_buffer(nbytes)
    written = _blosc_library().blosc_decompress(ctypes.cast(source, ctypes.c_void_p), dest, nbytes)
    if written < 0:
        raise RuntimeError("blosc decompress failed")
    return dest.raw[:written]


def _write_videos(store: _Store, keys: set[str], start: int, end: int, work: Path, fps: float) -> dict[str, Path]:
    videos = {}
    rate = max(1, int(round(fps)))
    for feature, candidates in _SCENE_KEYS:
        key = next((item for item in candidates if item in keys), None)
        if key is None:
            continue
        path = work / f"{feature.replace('.', '_')}.mp4"
        _write_mp4(_frames(store, f"data/{key}", start, end), path, fps=rate)
        videos[feature] = path
    finger_videos = _finger_videos(store, keys, start, end, work, rate)
    videos.update(finger_videos)
    if GRIPPER_IMAGE_LEFT not in videos and GRIPPER_IMAGE_RIGHT not in videos:
        raise RuntimeError("ViTaMIn episode has no image tactile arrays")
    return videos


def _finger_videos(store: _Store, keys: set[str], start: int, end: int, work: Path, fps: int) -> dict[str, Path]:
    chosen: list[tuple[str, int | None]] = []
    for key in sorted(item for item in keys if "tactile_data" in item):
        if not _is_image_tactile(store, keys, key):
            continue
        meta = json.loads(store.read(f"data/{key}/.zarray"))
        shape = meta["shape"]
        if len(shape) == 5 and int(shape[-1]) in {1, 3, 4} and int(shape[1]) <= 8:
            chosen.extend((key, slot) for slot in range(int(shape[1])))
        elif len(shape) == 4 and int(shape[-1]) in {1, 3, 4}:
            chosen.append((key, None))
    if not chosen:
        return {}
    slots: dict[str, tuple[str, int | None]] = {}
    stacked = [(key, slot) for key, slot in chosen if slot is not None]
    if stacked:
        slots[GRIPPER_IMAGE_LEFT] = stacked[0]
        if len(stacked) > 1:
            slots[GRIPPER_IMAGE_RIGHT] = stacked[1]
    else:
        slots[GRIPPER_IMAGE_LEFT] = chosen[0]
        if len(chosen) > 1:
            slots[GRIPPER_IMAGE_RIGHT] = chosen[1]
    videos = {}
    for feature, (key, slot) in slots.items():
        path = work / f"{feature.replace('.', '_')}.mp4"
        _write_mp4(_frames(store, f"data/{key}", start, end, slot=slot), path, fps=fps)
        videos[feature] = path
    return videos


def _is_image_tactile(store: _Store, keys: set[str], key: str) -> bool:
    type_key = key.replace("tactile_data", "tactile_type", 1)
    if type_key not in keys:
        return True
    values = _read_array(store, f"data/{type_key}", 0, 1).reshape(-1)
    if values.size == 0:
        return False
    text = _text(values[0]).lower()
    return text in {"", "image"}


def _frames(store: _Store, rel: str, start: int, end: int, slot: int | None = None):
    meta = json.loads(store.read(f"{rel}/.zarray"))
    chunks = tuple(int(item) for item in meta["chunks"])
    separator = meta.get("dimension_separator") or "."
    shape = tuple(int(item) for item in meta["shape"])
    counts = [(shape[axis] + chunks[axis] - 1) // chunks[axis] for axis in range(len(shape))]
    cached_id = None
    cached = None
    for index in range(start, end):
        chunk0 = index // chunks[0]
        if chunk0 != cached_id:
            cached = _read_chunk(store, rel, meta, (chunk0,), counts, separator)
            cached_id = chunk0
        local = index - chunk0 * chunks[0]
        frame = cached[local]
        if slot is not None:
            frame = frame[slot]
        yield _rgb_frame(frame)


def _rgb_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 1 or (array.ndim == 2 and array.shape[-1] not in {1, 3, 4}):
        raw = np.ascontiguousarray(array).tobytes()
        start = raw.find(b"\xff\xd8")
        if start >= 0:
            image = Image.open(io.BytesIO(raw[start:])).convert("RGB")
            return np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _write_mp4(frames, path: Path, fps: int) -> None:
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(f"no frames for {path}") from exc
    height, width = first.shape[:2]
    height -= height % 2
    width -= width % 2
    if height < 2 or width < 2:
        raise RuntimeError(f"frame is too small: {first.shape}")
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        for frame in itertools.chain((first,), iterator):
            proc.stdin.write(np.ascontiguousarray(frame[:height, :width]).tobytes())
    finally:
        proc.stdin.close()
    error = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {error[-400:]}")


def _columns(store: _Store, keys: set[str], start: int, end: int) -> dict[str, np.ndarray]:
    columns = {}
    pose_key = next((key for key in ("right_wrist_pose", "left_wrist_pose", "camera_ego_pose") if key in keys), None)
    if pose_key is not None:
        pose = _read_array(store, f"data/{pose_key}", start, end).astype(np.float32)
        euler = _rotvec_to_euler(pose[:, 3:6])
        ee_pose = np.concatenate([pose[:, :3], euler], axis=1).astype(np.float32)
        columns["observation.ee_pose"] = ee_pose
        columns["observation.ee_position"] = pose[:, :3]
        columns["action.ee_pose"] = np.vstack([ee_pose[1:], ee_pose[-1:]])
    joint_key = next((key for key in ("right_hand_joints", "left_hand_joints") if key in keys), None)
    if joint_key is not None:
        joints = _read_array(store, f"data/{joint_key}", start, end).astype(np.float32)
        if joints.ndim == 1:
            joints = joints.reshape(-1, 1)
        gripper = joints[:, :1]
        columns["observation.gripper_position"] = gripper
        columns["action.gripper_position"] = np.vstack([gripper[1:], gripper[-1:]])
    return columns


def _rotvec_to_euler(rotvec: np.ndarray) -> np.ndarray:
    euler = np.zeros_like(rotvec, dtype=np.float64)
    for index, vector in enumerate(rotvec):
        angle = float(np.linalg.norm(vector))
        if angle < 1e-8:
            continue
        axis = vector / angle
        x, y, z = (float(item) for item in axis)
        cosine, sine = float(np.cos(angle)), float(np.sin(angle))
        one_minus = 1.0 - cosine
        matrix = np.array(
            [
                [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine],
                [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine],
                [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus],
            ]
        )
        euler[index] = matrix_to_rpy(matrix)
    return euler.astype(np.float32)


def _fps(store: _Store, keys: set[str], start: int, end: int) -> float:
    if "timestamps" not in keys:
        return 30.0
    stamps = _read_array(store, "data/timestamps", start, end).astype(np.float64).reshape(-1)
    gaps = np.diff(stamps)
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if len(gaps) == 0:
        return 30.0
    gap = float(np.median(gaps))
    if gap > 1e6:
        return 1e9 / gap
    if gap > 10:
        return 1000.0 / gap
    # ViTaMIn stores a frame index here, one count apart. The cameras run at 30 Hz.
    if 0.8 <= gap <= 1.2:
        return 30.0
    return 1.0 / gap


def _task(store: _Store, keys: set[str], label: str, start: int, end: int) -> str:
    if "sub_task_instruction" in keys:
        values = _read_array(store, "data/sub_task_instruction", start, min(start + 1, end)).reshape(-1)
        if values.size:
            text = _text(values[0]).strip()
            if text:
                return text
    name = Path(label).name
    if name.endswith(".zarr"):
        name = name[: -len(".zarr")]
    return name.removesuffix("_dataset").replace("_", " ")


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return _text(value.item()) if value.shape == () else ""
    return str(value)
