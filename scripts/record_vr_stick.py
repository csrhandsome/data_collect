"""Record ros2 topic echo output without importing or starting robot components."""

import argparse
import csv
import json
import math
import os
import selectors
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

AXES = ("right_stick_x", "right_stick_y", "gripper_velocity_axis")


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是有限正数")
    return number


def main():
    parser = argparse.ArgumentParser(description="只订阅已有 VR 话题，记录右摇杆数值、范围和相邻跳变。")
    parser.add_argument("--topic", default="/vr/input", help="std_msgs/msg/String 话题（默认 /vr/input）")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/vr_stick",
                        help="记录父目录（默认仓库 data/vr_stick，每次新建子目录）")
    parser.add_argument("--duration", type=positive, help="录制秒数；默认持续到 Ctrl+C")
    parser.add_argument("--jump-threshold", type=positive, default=0.2,
                        help="相邻新采样绝对差值超过此值时标记（默认 0.2）")
    args = parser.parse_args()
    if not shutil.which("ros2"):
        parser.exit(1, "缺少 ROS2 CLI；请先进入已加载 ROS 环境的容器，再运行此脚本。\n")

    # A fresh directory prevents accidental overwrites, including concurrent runs.
    directory = args.output_dir.expanduser().resolve() / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}"
    )
    directory.mkdir(parents=True)
    command = ["ros2", "topic", "echo", args.topic, "std_msgs/msg/String",
               "--field", "data", "--full-length", "--no-daemon",
               "--qos-reliability", "reliable",
               "--qos-durability", "volatile", "--qos-depth", "1000"]
    stats = {axis: {"min": None, "max": None, "max_abs_delta": None, "jump_count": 0}
             for axis in AXES}
    summary = {
        "topic": args.topic, "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
        "command": command, "jump_threshold": args.jump_threshold,
        "started_at": datetime.now().astimezone().isoformat(),
        "messages": 0, "new_source_samples": 0, "repeated_messages": 0,
        "uninitialized_messages": 0, "invalid_lines": 0, "source_resets": 0,
        "max_source_dt_s": None, "axes": stats,
    }
    previous = None
    previous_values = None
    started = time.monotonic()
    next_display = started

    def receive(line, writer):
        nonlocal previous, previous_values, next_display
        if not line.strip() or line.strip() == b"---":
            return
        try:
            sample = json.loads(line)
            values = {axis: float(sample[axis]) for axis in AXES}
            seq, source_ns = int(sample["pose_seq"]), int(sample["pose_monotonic_ns"])
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("non-finite axis")
        except (ValueError, TypeError, KeyError, UnicodeError, OverflowError):
            summary["invalid_lines"] += 1
            if summary["invalid_lines"] <= 3:
                print("收到非预期行，已保留在 echo.log 中。", flush=True)
            return
        summary["messages"] += 1
        identity = (seq, source_ns)
        new_source = source_ns > 0 and seq > 0 and identity != previous
        deltas = {axis: None for axis in AXES}
        dt = None
        if source_ns <= 0 or seq <= 0:
            summary["uninitialized_messages"] += 1
        elif new_source:
            summary["new_source_samples"] += 1
            if previous is not None:
                if seq <= previous[0] or source_ns <= previous[1]:
                    summary["source_resets"] += 1
                else:
                    dt = (source_ns - previous[1]) / 1e9
                    summary["max_source_dt_s"] = max(summary["max_source_dt_s"] or 0, dt)
                    deltas = {axis: values[axis] - previous_values[axis] for axis in AXES}
            for axis, value in values.items():
                stat = stats[axis]
                stat["min"] = value if stat["min"] is None else min(stat["min"], value)
                stat["max"] = value if stat["max"] is None else max(stat["max"], value)
                delta = deltas[axis]
                if delta is not None:
                    stat["max_abs_delta"] = max(stat["max_abs_delta"] or 0, abs(delta))
                    stat["jump_count"] += abs(delta) > args.jump_threshold
            previous, previous_values = identity, values
        else:
            summary["repeated_messages"] += 1
        jumps = {axis: deltas[axis] is not None and abs(deltas[axis]) > args.jump_threshold
                 for axis in AXES}
        writer.writerow({
            "received_unix_ns": time.time_ns(), "elapsed_s": time.monotonic() - started,
            "pose_seq": seq, "pose_monotonic_ns": source_ns,
            "timestamp": sample.get("timestamp"), "new_source": int(new_source),
            "source_dt_s": dt, **values,
            **{f"delta_{axis}": delta for axis, delta in deltas.items()},
            **{f"jump_{axis}": int(jump) for axis, jump in jumps.items()},
        })
        if time.monotonic() >= next_display or any(jumps.values()):
            labels = [f"{axis}={value:+.6f}" for axis, value in values.items()]
            print("  ".join(labels) + ("  [超过跳变阈值]" if any(jumps.values()) else ""), flush=True)
            next_display = time.monotonic() + 0.2

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    fields = ["received_unix_ns", "elapsed_s", "pose_seq", "pose_monotonic_ns", "timestamp",
              "new_source", "source_dt_s", *AXES,
              *(f"delta_{axis}" for axis in AXES), *(f"jump_{axis}" for axis in AXES)]
    print(f"只订阅 {args.topic}；ROS_DOMAIN_ID={summary['ros_domain_id']}\n保存到 {directory}")
    print("请缓慢推动右摇杆、停在两端、再松手回中。Ctrl+C 结束并生成摘要。")
    process = None
    status = 0
    try:
        with (directory / "echo.log").open("wb", buffering=0) as raw, \
                (directory / "samples.csv").open("w", newline="", buffering=1) as table, \
                (directory / "ros2.stderr.log").open("wb", buffering=0) as errors, \
                selectors.DefaultSelector() as selector:
            writer = csv.DictWriter(table, fieldnames=fields)
            writer.writeheader()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors,
                                       env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                       start_new_session=True)
            selector.register(process.stdout, selectors.EVENT_READ)
            pending = b""
            while True:
                remaining = None if args.duration is None else args.duration - (time.monotonic() - started)
                if remaining is not None and remaining <= 0:
                    summary["stop_reason"] = "duration"
                    break
                if not selector.select(timeout=min(0.2, remaining) if remaining is not None else 0.2):
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    if pending:
                        receive(pending, writer)
                    status = process.wait()
                    summary["stop_reason"] = "echo_exited"
                    break
                raw.write(chunk)
                pending += chunk
                lines = pending.split(b"\n")
                pending = lines.pop()
                for line in lines:
                    receive(line, writer)
    except KeyboardInterrupt:
        summary["stop_reason"] = "signal"
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # Signal only our CLI session; independent VR/robot processes are untouched.
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
            summary["echo_returncode"] = process.returncode
        summary["elapsed_s"] = time.monotonic() - started
        summary["ended_at"] = datetime.now().astimezone().isoformat()
        (directory / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        print(f"\n收到 {summary['messages']} 条消息，{summary['new_source_samples']} 个新采样，"
              f"{summary['repeated_messages']} 条重复状态。")
        for axis, stat in stats.items():
            print(f"{axis}: min={stat['min']}, max={stat['max']}, "
                  f"max|Δ|={stat['max_abs_delta']}, 超阈值次数={stat['jump_count']}")
        print(f"已保存：{directory}")
    if summary["new_source_samples"] == 0:
        print("未收到有效头显采样；请检查 VR 连接、话题和 ROS_DOMAIN_ID。")
        status = status or 1
    if status:
        print(f"记录结束状态：{status}；错误日志：{directory / 'ros2.stderr.log'}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
