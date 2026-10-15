"""Shared confined lookup for historical audio/ and new root-level sidecars."""

import fcntl
import json
import os
import tempfile
from pathlib import Path


def confined(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Sidecar path escapes dataset root")
    return path


def read_sync(root, episode_index):
    for folder in ["audio", ""]:
        path = confined(root, f"{folder}/episode_{episode_index:06d}.sync.json".lstrip("/"))
        if path.is_file():
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict) or payload.get("episode_index") != episode_index:
                raise ValueError("Sync sidecar episode mismatch")
            return payload
    return {}


def update_sync(root, episode_index, changes, *, audio=False):
    """Merge sidecar fields under a process lock, publishing JSON atomically.

    Collection, annotations and offline audio processing share sidecars. Merge
    only changed fields so a slow audio job cannot overwrite a newer annotation.
    """
    root = Path(root).resolve()
    lock_path = root / ".episode-sync.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    temporary = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        sync = read_sync(root, episode_index)
        name = f"episode_{episode_index:06d}.sync.json"
        relative = Path("audio" if audio else "") / name
        for folder in ["audio", ""]:
            candidate = root / folder / name
            if candidate.exists() or candidate.is_symlink():
                relative = Path(folder) / name
                break
        path = root / relative
        if path.resolve() != path:
            raise ValueError("Cannot write a symlinked sidecar")
        sync.update(changes)
        sync["episode_index"] = episode_index
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".sync.tmp", delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump(sync, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
        return sync
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        os.close(descriptor)


def origin_ns(sync):
    frames = sync.get("frame_records", [])
    return (
        int(frames[0]["host_frame_monotonic_ns"])
        if frames
        else int(sync.get("episode_start_monotonic_ns", 0))
    )


def catalog(root, info, episode_index):
    from ._common import _feature_blocks

    blocks = _feature_blocks(info)
    sync = read_sync(root, episode_index)
    records = sync.get("frame_records", [])
    keys = {block["key"] for block in blocks}
    if records:
        for key, label, shape, names in [
            ("ee_position", "末端位置", [3], ["x", "y", "z"]),
            ("ee_orientation_xyzw", "末端四元数", [4], ["qx", "qy", "qz", "qw"]),
        ]:
            if key in records[0] and key not in keys:
                blocks.append(
                    {
                        "key": key,
                        "label": label,
                        "kind": "ee" if shape == [3] else "series",
                        "dtype": "float32",
                        "shape": shape,
                        "names": names,
                    }
                )
    audio_path = sync.get("audio_path", f"audio/episode_{episode_index:06d}.wav")
    if confined(root, audio_path).is_file():
        blocks.append(
            {
                "key": "audio",
                "label": "麦克风音频",
                "kind": "audio",
                "dtype": "wav",
                "shape": [1],
                "names": None,
            }
        )
    trace = sync.get("action_trace")
    if trace and confined(root, trace).is_file():
        for key, label, size in [
            ("trace_joint_position", "100 Hz 关节状态", 7),
            ("trace_ee_position", "100 Hz 末端位置", 3),
            ("trace_vr_position", "100 Hz VR 重采样", 3),
            ("trace_target_ee_position", "100 Hz EE 目标", 3),
        ]:
            blocks.append(
                {
                    "key": key,
                    "label": label,
                    "kind": "series",
                    "dtype": "float64",
                    "shape": [size],
                    "names": None,
                }
            )
    return blocks
