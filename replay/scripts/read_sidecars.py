"""Shared confined lookup for historical audio/ and new root-level sidecars."""

import json
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
