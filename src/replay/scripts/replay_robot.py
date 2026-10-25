"""Replay one saved LeRobot episode through the public Panda controller.

Run: uv run python -m replay.scripts.replay_robot --dataset <root> [--dry-run]
Without --episode-index, use the last saved episode. Never opens cameras or VR.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from replay.scripts._common import _episode_table, _get_episode, _load_metadata, _safe_path
from replay.scripts.read_sidecars import read_sync


def load_trajectory(root: Path, episode_index: int | None = None) -> tuple[int, list[dict]]:
    """Load and validate the entire trajectory before connecting to hardware."""
    root = root.resolve()
    info, episodes = _load_metadata(root)
    if not episodes:
        raise ValueError("数据集没有已保存的 episode")
    index = max(episodes) if episode_index is None else episode_index
    episode = _get_episode(episodes, index)
    sync = read_sync(root, index)
    trace = sync.get("action_trace")
    if trace:
        rows = [
            json.loads(line)
            for line in _safe_path(root, trace).read_text().splitlines()
            if line.strip()
        ]
        clock = "host_sample_monotonic_ns"
        scale = 1e9
    elif sync.get("frame_records"):
        rows = sync["frame_records"]
        clock = "host_frame_monotonic_ns"
        scale = 1e9
    else:
        key = next(
            (
                key
                for key in (
                    "observation.ee_pose",
                    "observation.joint_position",
                    "ee_pose",
                    "joint_position",
                )
                if key in info["features"]
            ),
            None,
        )
        if key is None:
            raise ValueError("回放需要完整 ee_pose 或 joint_position，只有 XYZ 位置不足以回放")
        rows = _episode_table(root, info, episode, key).to_pylist()
        grip_key = next(
            (
                key
                for key in ("observation.gripper_position", "gripper_position")
                if key in info["features"]
            ),
            None,
        )
        if grip_key is not None:
            gripper = _episode_table(root, info, episode, grip_key)[grip_key].to_pylist()
            for row, ratio in zip(rows, gripper, strict=True):
                row["gripper_position"] = ratio
        rows = [
            {key.removeprefix("observation."): value for key, value in row.items()} for row in rows
        ]
        clock, scale = "timestamp", 1
    if not rows:
        raise ValueError("轨迹为空")
    from control.util.pose import normalize_quat_xyzw

    first = float(rows[0][clock])
    result = []
    for row in rows:
        point = {"time_s": (float(row[clock]) - first) / scale}
        if "ee_position" in row and "ee_orientation_xyzw" in row:
            point["position"] = np.asarray(row["ee_position"], dtype=float)
            point["quaternion_xyzw"] = normalize_quat_xyzw(row["ee_orientation_xyzw"])
        elif "ee_pose" in row:
            from control.util.pose import pose6_to_quat_xyzw

            pose = np.asarray(row["ee_pose"], dtype=float)
            if pose.shape != (6,) or not np.isfinite(pose).all():
                raise ValueError("ee_pose 必须是有限的 XYZ + RPY 六维向量")
            point["position"] = pose[:3]
            point["quaternion_xyzw"] = pose6_to_quat_xyzw(pose)
        if "joint_position" in row:
            point["joints"] = np.asarray(row["joint_position"], dtype=float)
            if point["joints"].shape != (7,) or not np.isfinite(point["joints"]).all():
                raise ValueError("joint_position 必须是有限的七维向量")
        if "position" not in point and "joints" not in point:
            raise ValueError("轨迹缺少完整位姿或关节记录")
        if "position" in point and (
            point["position"].shape != (3,) or not np.isfinite(point["position"]).all()
        ):
            raise ValueError("末端位置无效")
        if "gripper_position" in row:
            ratio = np.asarray(row["gripper_position"], dtype=float).reshape(-1)
            if len(ratio) != 1 or not np.isfinite(ratio).all() or not 0 <= ratio[0] <= 1:
                raise ValueError("夹爪开合比例必须在 0..1 范围内")
            point["gripper"] = float(ratio[0])
        result.append(point)
    times = np.asarray([point["time_s"] for point in result])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("轨迹时间必须有限且严格递增")
    return index, result


def replay_trajectory(arm, points, *, speed=1.0):
    """EE streaming with recorded timing; gripper moves pause the playback clock."""
    from control.util.pose import slerp_quat_xyzw
    from control.util.robot import finish_stream

    if not np.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("回放速度必须在 (0, 1] 范围内")
    # Joint-only historical data is converted through this controller's kinematics.
    points = [dict(point) for point in points]
    for point in points:
        if "position" not in point:
            pose = arm.pose_from_joints(point["joints"])
            point.update(position=pose.xyz_m, quaternion_xyzw=pose.quaternion_xyzw())
    first = points[0]
    print("正在移动至 episode 起始姿态…", flush=True)
    if "joints" in first:
        arm.move_joints(first["joints"], speed_factor=0.1)
    arm.move_ee(first["position"], first["quaternion_xyzw"], speed_factor=0.1)
    previous_ratio = None
    has_gripper = arm.get_capabilities().gripper
    times = np.asarray([point["time_s"] for point in points])
    arm.start_stream()
    started = time.monotonic()
    last_report = -1
    gripper_cursor = 0
    try:
        while True:
            elapsed = min((time.monotonic() - started) * speed, times[-1])
            i = min(int(np.searchsorted(times, elapsed, side="right")) - 1, len(points) - 1)
            point = points[i]
            # Preserve discrete gripper events even if a scheduling tick skips samples.
            while gripper_cursor <= i:
                ratio = points[gripper_cursor].get("gripper")
                if (
                    has_gripper
                    and ratio is not None
                    and (previous_ratio is None or abs(ratio - previous_ratio) >= 0.01)
                ):
                    before = time.monotonic()
                    arm.set_gripper(ratio, wait=True, timeout=10)
                    started += time.monotonic() - before
                    previous_ratio = ratio
                gripper_cursor += 1
            position, quaternion = point["position"], point["quaternion_xyzw"]
            if i + 1 < len(points):
                following = points[i + 1]
                fraction = (elapsed - times[i]) / (times[i + 1] - times[i])
                position = position + fraction * (following["position"] - position)
                quaternion = slerp_quat_xyzw(quaternion, following["quaternion_xyzw"], fraction)
            arm.send_ee_target(position, quaternion)
            second = int(elapsed)
            if second != last_report:
                print(f"回放 {elapsed:.1f} / {times[-1]:.1f} 秒", flush=True)
                last_report = second
            if elapsed >= times[-1]:
                break
            time.sleep(0.01)
        # Keep the final target until its measured pose arrives, then stop at that pose.
        from control.util.pose import quat_angle_xyzw

        deadline = time.monotonic() + 3
        while True:
            state = arm.get_state()
            if (
                np.linalg.norm(state.ee_position - points[-1]["position"]) <= 0.01
                and quat_angle_xyzw(state.ee_quaternion_xyzw, points[-1]["quaternion_xyzw"]) <= 0.05
            ):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("回放末端未在三秒内到达目标")
            time.sleep(0.01)
    finally:
        finish_stream(arm)


def main(argv=None):
    from control.config import DEFAULT_CONFIG, load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite(args.speed) or not 0 < args.speed <= 1:
        parser.error("--speed 必须在 (0, 1] 范围内")
    if args.episode_index is not None and args.episode_index < 0:
        parser.error("--episode-index 必须 >= 0")
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    index, points = load_trajectory(args.dataset, args.episode_index)
    print(
        f"Episode {index:06d} · {len(points)} 个采样 · {'模拟' if args.dry_run else '真机'}",
        flush=True,
    )
    from control._panda.fake import FakeBackend
    from control.robotic_arm_controller import RoboticArmControler

    try:
        with RoboticArmControler(
            config=config, backend=FakeBackend() if args.dry_run else None
        ) as arm:
            replay_trajectory(arm, points, speed=args.speed)
    except KeyboardInterrupt:
        print("回放已停止，机械臂连接已关闭。", flush=True)
        return 0
    print("回放完成。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
