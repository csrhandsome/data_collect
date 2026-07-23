#!/usr/bin/env python3
"""
Replay or inspect LeRobot joint-position datasets collected for OpenPI, with audio.

Default behavior: print summary, show recorded images, and replay the aligned
episode audio segment.
Use --execute to actually send recorded joint-position actions to the robot.

uv run -m data_analysis.replay_openpi_lerobot_joint_audio
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from control.robotic_arm_controller import RoboticArmControler
from control.util.audio_util import (
    load_episode_audio_segment,
    start_audio_playback,
    stop_audio_playback,
)
from control.util.img_util import decode_image, show_image
from data_analysis.replay_openpi_lerobot_joint import (
    _as_int,
    _as_text,
    _can_show_images,
    _default_data_root,
    _find_latest_repo_id,
    _get_episode_indices,
    _list_episodes,
    _load_dataset,
    _sleep_until,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay/inspect LeRobot dataset (OpenPI joint-position actions + audio)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="LeRobot repo id. If omitted, use the newest dataset under data/openpi.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional dataset root (defaults to ./data/<repo_id>)",
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
    parser.add_argument("--mute", action="store_true", help="Disable audio playback")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute joint-position actions on the robot",
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
    show_images = not args.no_show
    if show_images and not _can_show_images():
        show_images = False
        print(
            "[Replay] Warning: no GUI display detected; disabling image windows. "
            "Use --no-show to suppress this warning."
        )

    repo_id = args.repo_id
    if not repo_id:
        openpi_root = _default_data_root() / "openpi"
        repo_id = _find_latest_repo_id(openpi_root, prefix="franka_lerobot_")
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

    info = json.loads((ds_root / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 15.0))
    episode_prompt = _as_text(ds[int(ep_indices[0])].get("task"))

    audio_segment = None
    if not args.mute:
        audio_segment = load_episode_audio_segment(
            ds_root=ds_root,
            episode_index=episode_index,
            start_frame_index=start,
            end_frame_index=end,
            fps=fps,
        )

    print("=" * 70)
    print("LeRobot replay (joint position + audio)")
    print("=" * 70)
    print(f"Dataset: {repo_id}")
    print(f"Root: {ds_root}")
    print(f"Episode: {episode_index}")
    print(f"Frames: {len(ep_indices)} (start={start}, end={end})")
    print(f"FPS: {fps:.2f}")
    print(f"Speed: {args.speed}x")
    print(f"Audio: {'muted' if args.mute else 'enabled'}")
    if audio_segment is not None:
        print(
            "Audio clip: "
            f"{audio_segment.audio_path} "
            f"({audio_segment.duration_s:.2f}s, "
            f"samples {audio_segment.start_sample}:{audio_segment.end_sample})"
        )
    elif not args.mute:
        print("Audio clip: unavailable")
    print(f"Execute: {args.execute}")
    if episode_prompt:
        print(f"Prompt: {episode_prompt}")
    print("=" * 70)

    arm = None
    action_freq = float(fps * args.speed)
    if args.execute:
        from panda_py import controllers

        arm = RoboticArmControler()
        first_action = np.asarray(ds[int(ep_indices[0])]["actions"], dtype=np.float64)
        if not args.no_init:
            print("Moving to first recorded joint target...")
            arm.panda.move_to_joint_position(first_action[:7])
            if not arm.wait_until_stopped():
                max_vel = float(np.max(np.abs(np.asarray(arm.panda.get_state().dq))))
                print(
                    "[Warning] Robot did not fully stop after move_to_joint_position: "
                    f"max_vel={max_vel:.4f} rad/s"
                )

    try:
        for loop_idx in range(int(args.loop)):
            print(f"\n[Replay] Loop {loop_idx + 1}/{args.loop}")
            if args.execute and arm is not None:
                start_audio_playback(audio_segment, speed=float(args.speed))
                ctrl = controllers.JointPosition()
                arm.panda.start_controller(ctrl)
                with arm.panda.create_context(frequency=action_freq) as ctx:
                    last_gripper_open = None
                    for frame_idx in ep_indices:
                        if not ctx.ok():
                            break
                        item = ds[int(frame_idx)]
                        action = np.asarray(item["actions"], dtype=np.float32)
                        prompt = _as_text(item.get("task"))
                        joint_target = np.asarray(action[:7], dtype=np.float64)
                        gripper_cmd = float(action[7])
                        gripper_open = gripper_cmd > float(args.gripper_threshold)

                        if (
                            last_gripper_open is None
                            or gripper_open != last_gripper_open
                        ):
                            arm.panda.stop_controller()
                            if gripper_open:
                                arm.gripper_open()
                            else:
                                arm.gripper_close()
                            ctrl = controllers.JointPosition()
                            arm.panda.start_controller(ctrl)
                            last_gripper_open = gripper_open

                        ctrl.set_control(joint_target, np.zeros(7, dtype=np.float64))

                        if show_images:
                            ep_idx = _as_int(item.get("episode_index"))
                            fr_idx = _as_int(item.get("frame_index"))
                            ext = decode_image(
                                item.get("exterior_image_1_left"),
                                root=ds_root,
                                image_key="exterior_image_1_left",
                                episode_index=ep_idx,
                                frame_index=fr_idx,
                            )
                            wrist = decode_image(
                                item.get("wrist_image_left"),
                                root=ds_root,
                                image_key="wrist_image_left",
                                episode_index=ep_idx,
                                frame_index=fr_idx,
                            )
                            show_image(ext, win_name="Recorded External", prompt=prompt)
                            show_image(wrist, win_name="Recorded Wrist", prompt=prompt)
            else:
                dt = (1.0 / fps) / float(args.speed)
                loop_start_time = time.monotonic()
                start_audio_playback(audio_segment, speed=float(args.speed))
                for offset, frame_idx in enumerate(ep_indices):
                    _sleep_until(loop_start_time + (offset * dt))
                    item = ds[int(frame_idx)]
                    prompt = _as_text(item.get("task"))
                    if show_images:
                        ep_idx = _as_int(item.get("episode_index"))
                        fr_idx = _as_int(item.get("frame_index"))
                        ext = decode_image(
                            item.get("exterior_image_1_left"),
                            root=ds_root,
                            image_key="exterior_image_1_left",
                            episode_index=ep_idx,
                            frame_index=fr_idx,
                        )
                        wrist = decode_image(
                            item.get("wrist_image_left"),
                            root=ds_root,
                            image_key="wrist_image_left",
                            episode_index=ep_idx,
                            frame_index=fr_idx,
                        )
                        show_image(ext, win_name="Recorded External", prompt=prompt)
                        show_image(wrist, win_name="Recorded Wrist", prompt=prompt)
    except KeyboardInterrupt:
        print("\n[Replay] Interrupted")
    finally:
        stop_audio_playback()
        if args.execute and arm is not None:
            try:
                arm.panda.stop_controller()
            except Exception:
                pass
            arm.cleanup()


if __name__ == "__main__":
    main()
