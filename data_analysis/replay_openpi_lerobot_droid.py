#!/usr/bin/env python3
"""
Replay or inspect LeRobot datasets collected for OpenPI (DROID-style keys).

Default behavior: print summary and optionally show recorded images.
Use --execute to actually send joint-velocity commands to the robot.
夹爪抓的时候会导致速度控制器停止，视频也会停止一下
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from io import BytesIO
from pathlib import Path

import numpy as np

import datasets as _hf_datasets

from control.robotic_arm_controller import RoboticArmControler


def _default_data_root() -> Path:
    return Path(__file__).resolve().parent / "data"


def _default_dataset_root(repo_id: str) -> Path:
    return _default_data_root() / repo_id


def _find_latest_repo_id(data_root: Path, *, prefix: str | None = None) -> str:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    candidates = [p for p in data_root.iterdir() if p.is_dir()]
    if prefix:
        prefixed = [p for p in candidates if p.name.startswith(prefix)]
        if prefixed:
            candidates = prefixed

    if not candidates:
        raise FileNotFoundError(f"No datasets found under: {data_root}")

    latest_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest_dir.relative_to(_default_data_root()))


def _load_dataset(repo_id: str, root: str | None) -> tuple[_hf_datasets.Dataset, Path]:
    root_path = Path(root) if root else _default_dataset_root(repo_id)
    parquet_files = sorted(root_path.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {root_path / 'data'}")
    ds = _hf_datasets.Dataset.from_parquet([str(p) for p in parquet_files])
    return ds, root_path


def _get_episode_indices(ds: _hf_datasets.Dataset, episode_index: int) -> np.ndarray:
    ep_all = np.asarray(ds["episode_index"], dtype=np.int64)
    return np.flatnonzero(ep_all == int(episode_index))


def _list_episodes(ds: _hf_datasets.Dataset) -> list[int]:
    return sorted(set(ds["episode_index"]))


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            try:
                return _as_text(value.item())
            except Exception:
                return ""
        return ""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _as_text(value[0])
        return " ".join(part for part in (_as_text(v) for v in value) if part)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""
    return str(value).strip()


def _draw_prompt_overlay(bgr: np.ndarray, prompt: str) -> np.ndarray:
    prompt = prompt.strip()
    if not prompt:
        return bgr

    try:
        import cv2
    except Exception:
        return bgr

    annotated = bgr.copy()
    height, width = annotated.shape[:2]
    margin = max(8, width // 50)
    font_scale = max(0.45, min(0.8, width / 960.0))
    thickness = 1 if width < 960 else 2
    line_height = max(20, int(26 * font_scale))
    max_chars = max(24, width // 12)
    lines = textwrap.wrap(f"Prompt: {prompt}", width=max_chars)[:4]
    box_height = margin * 2 + line_height * len(lines)

    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (margin, margin),
        (width - margin, min(height - margin, margin + box_height)),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0.0, annotated)

    y = margin + line_height
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (margin * 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height

    return annotated


def show_image(
    frame: np.ndarray,
    *,
    win_name: str,
    scale: float = 1.0,
    prompt: str = "",
) -> None:
    try:
        import cv2
    except Exception:
        return
    if frame is None:
        return
    img = np.asarray(frame)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
    if img.ndim == 3 and img.shape[-1] == 3:
        bgr = img[..., ::-1]
        bgr = _draw_prompt_overlay(bgr, prompt)
    else:
        return
    if scale != 1.0:
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    cv2.imshow(win_name, bgr)
    cv2.waitKey(1)


def _as_int(value: object) -> int | None:
    if isinstance(value, np.ndarray):
        try:
            return int(value.item())
        except Exception:
            return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _decode_image(
    value: object,
    *,
    root: Path,
    image_key: str,
    episode_index: int | None,
    frame_index: int | None,
) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    try:
        from PIL import Image
    except Exception:
        return None

    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))

    if isinstance(value, (bytes, bytearray)):
        try:
            return np.asarray(Image.open(BytesIO(value)).convert("RGB"))
        except Exception:
            return None

    if isinstance(value, dict):
        bytes_data = value.get("bytes")
        if bytes_data:
            try:
                return np.asarray(Image.open(BytesIO(bytes_data)).convert("RGB"))
            except Exception:
                return None
        path = value.get("path")
        if path:
            candidate = Path(path)
            if not candidate.is_absolute():
                if episode_index is not None and frame_index is not None:
                    candidate = (
                        root
                        / "images"
                        / image_key
                        / f"episode-{episode_index:06d}"
                        / Path(path).name
                    )
                else:
                    candidate = root / Path(path)
            if candidate.exists():
                try:
                    return np.asarray(Image.open(candidate).convert("RGB"))
                except Exception:
                    return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay/inspect LeRobot dataset (OpenPI DROID-style keys)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="LeRobot repo id (e.g. openpi/franka_droid_lerobot_YYYYmmdd_HHMMSS). "
        "If omitted, use the newest dataset under data/openpi.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional dataset root (defaults to ./data/<repo_id>) ",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to replay (default: latest)",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start frame index within the episode"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="End frame index within the episode"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier (1.0 = real time)",
    )
    parser.add_argument("--loop", type=int, default=1, help="Number of loops")
    parser.add_argument("--no-show", action="store_true", help="Hide recorded images")
    parser.add_argument(
        "--execute", action="store_true", help="Execute joint velocities on the robot"
    )
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=1.0,
        help="Scale joint velocity commands",
    )
    parser.add_argument(
        "--gripper-threshold", type=float, default=0.5, help="Gripper open threshold"
    )
    parser.add_argument(
        "--no-init", action="store_true", help="Skip moving to start position"
    )

    args = parser.parse_args()

    repo_id = args.repo_id
    if not repo_id:
        openpi_root = _default_data_root() / "openpi"
        repo_id = _find_latest_repo_id(openpi_root, prefix="franka_droid_lerobot_")
        print(f"Auto-selected latest dataset: {repo_id}")

    ds, ds_root = _load_dataset(repo_id, args.root)
    episodes = _list_episodes(ds)
    if not episodes:
        print("No episodes found in dataset.")
        sys.exit(1)

    episode_index = episodes[-1] if args.episode is None else args.episode
    if episode_index not in episodes:
        print(f"Episode {episode_index} not found. Available: {episodes}")
        sys.exit(1)

    ep_indices = _get_episode_indices(ds, episode_index)
    if ep_indices.size == 0:
        print(f"No frames found for episode {episode_index}")
        sys.exit(1)

    start = max(0, int(args.start))
    end = int(args.end) if args.end is not None else ep_indices.size
    end = min(end, ep_indices.size)
    ep_indices = ep_indices[start:end]
    if ep_indices.size == 0:
        print("Empty frame range after applying start/end.")
        sys.exit(1)

    # Read fps from meta/info.json
    import json

    _info = json.loads((ds_root / "meta" / "info.json").read_text())
    fps = float(_info.get("fps", 15.0))
    episode_prompt = _as_text(ds[int(ep_indices[0])].get("task"))
    print("=" * 70)
    print("LeRobot replay")
    print("=" * 70)
    print(f"Dataset: {repo_id}")
    print(f"Root: {ds_root}")
    print(f"Episode: {episode_index}")
    print(f"Frames: {len(ep_indices)} (start={start}, end={end})")
    print(f"FPS: {fps:.2f}")
    print(f"Speed: {args.speed}x")
    print(f"Execute: {args.execute}")
    if episode_prompt:
        print(f"Prompt: {episode_prompt}")
    print("=" * 70)

    arm = None
    if args.execute:
        arm = RoboticArmControler()
        if not args.no_init:
            arm.move_to_start()
        action_freq = float(fps * args.speed)
        control_freq = max(action_freq * 10.0, 200.0)
        MAX_JOINT_DELTA = 0.2  # rad/step, matches DROID
        arm.start_velocity_streaming(
            control_frequency=control_freq, time_step=1.0 / action_freq
        )

    try:
        for loop_idx in range(int(args.loop)):
            print(f"\n[Replay] Loop {loop_idx + 1}/{args.loop}")
            # Using create_context only when executing, to maintain control frequency.
            if args.execute and arm is not None:
                with arm.panda.create_context(frequency=action_freq) as ctx:
                    last_gripper_open = None
                    for frame_idx in ep_indices:
                        if not ctx.ok():
                            break
                        item = ds[int(frame_idx)]
                        action = np.asarray(item["actions"], dtype=np.float32)
                        prompt = _as_text(item.get("task"))
                        # action[:7] is normalized [-1,1]; convert to rad/s for apply_joint_velocity
                        joint_vel = (
                            action[:7]
                            * MAX_JOINT_DELTA
                            * action_freq
                            * float(args.velocity_scale)
                        )
                        gripper_cmd = float(action[7])
                        gripper_open = gripper_cmd > float(args.gripper_threshold)

                        if (
                            last_gripper_open is None
                            or gripper_open != last_gripper_open
                        ):
                            arm.stop_velocity_streaming()
                            if gripper_open:
                                arm.gripper_open()
                            else:
                                arm.gripper_close()
                            arm.start_velocity_streaming(
                                control_frequency=control_freq,
                                time_step=1.0 / action_freq,
                            )
                            last_gripper_open = gripper_open

                        arm.apply_joint_velocity(joint_vel, streaming=True)

                        if not args.no_show:
                            ep_idx = _as_int(item.get("episode_index"))
                            fr_idx = _as_int(item.get("frame_index"))
                            ext = _decode_image(
                                item.get("exterior_image_1_left"),
                                root=ds_root,
                                image_key="exterior_image_1_left",
                                episode_index=ep_idx,
                                frame_index=fr_idx,
                            )
                            wrist = _decode_image(
                                item.get("wrist_image_left"),
                                root=ds_root,
                                image_key="wrist_image_left",
                                episode_index=ep_idx,
                                frame_index=fr_idx,
                            )
                            show_image(
                                ext,
                                win_name="Recorded External",
                                prompt=prompt,
                            )
                            show_image(
                                wrist,
                                win_name="Recorded Wrist",
                                prompt=prompt,
                            )
            else:
                dt = (1.0 / fps) / float(args.speed)
                for frame_idx in ep_indices:
                    item = ds[int(frame_idx)]
                    action = np.asarray(item["actions"], dtype=np.float32)
                    prompt = _as_text(item.get("task"))
                    if not args.no_show:
                        ep_idx = _as_int(item.get("episode_index"))
                        fr_idx = _as_int(item.get("frame_index"))
                        ext = _decode_image(
                            item.get("exterior_image_1_left"),
                            root=ds_root,
                            image_key="exterior_image_1_left",
                            episode_index=ep_idx,
                            frame_index=fr_idx,
                        )
                        wrist = _decode_image(
                            item.get("wrist_image_left"),
                            root=ds_root,
                            image_key="wrist_image_left",
                            episode_index=ep_idx,
                            frame_index=fr_idx,
                        )
                        show_image(ext, win_name="Recorded External", prompt=prompt)
                        show_image(wrist, win_name="Recorded Wrist", prompt=prompt)
                    time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[Replay] Interrupted")
    finally:
        if args.execute and arm is not None:
            try:
                arm.stop_velocity_streaming()
            except Exception:
                pass
            arm.cleanup()


if __name__ == "__main__":
    main()
