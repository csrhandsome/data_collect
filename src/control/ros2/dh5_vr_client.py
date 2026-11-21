"""Forward live VR input to the standalone DH5 node while the client runs."""

import argparse
import csv
import json
import os
import signal
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String, UInt16

from control.yaml_config import DEFAULT_CONFIG, load_yaml_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", action="store_true", help="显示控制反馈并保存 CSV")
    parser.add_argument("--monitor-only", action="store_true", help="只监控正式 runner 的 DH5 控制链路")
    args, ros_args = parser.parse_known_args(argv)
    config = load_yaml_config(os.environ.get("DATA_COLLECT_CONFIG", DEFAULT_CONFIG))
    topic = config.get("vr", {}).get("ros2", {}).get("topic", "/vr/input")
    command_topic = "/dh5/vr_command"
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    rclpy.init(args=[sys.argv[0], *ros_args], signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = Node("dh5_vr_client")
        commands = None if args.monitor_only else node.create_publisher(String, command_topic, 1)
        with ExitStack() as stack:
            if args.monitor or args.monitor_only:
                monitor(node, commands, topic, stack)
            else:
                node.create_subscription(String, topic, commands.publish, 1)
            mode = "Monitoring only" if commands is None else f"Forwarding {topic} to {command_topic}"
            node.get_logger().info(f"{mode}. Ctrl+C to stop.")
            while not stopping:
                rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


def monitor(node, commands, topic, stack):
    fields = (
        "client_ns", "event", "pose_seq", "pose_monotonic_ns",
        "right_stick_x", "right_stick_y", "gripper_velocity_axis",
        "sample_ns", "received_seq", "received_axis", "applied_axis", "target_position",
        "actual_position", "grip_status", "velocity_setting", "feedback_error",
        "command_error", "control_state",
    )
    directory = Path(__file__).resolve().parents[3] / "data/dh5_vr"
    directory.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"{name}_{os.getpid()}.csv"
    stream = stack.enter_context(path.open("x", newline="", buffering=1))
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    print(f"DH5 监控记录保存到 {path}", flush=True)
    latest_vr = {}
    latest_state = {}
    last_state_ns = None

    def write(event, payload):
        writer.writerow({**payload, "client_ns": time.monotonic_ns(), "event": event})

    def forward(message):
        if commands is not None:
            commands.publish(message)
        payload = json.loads(message.data)
        if isinstance(payload, dict):
            latest_vr.update(payload)
            write("forwarded" if commands is not None else "input", payload)

    def receive_state(message):
        nonlocal last_state_ns
        payload = json.loads(message.data)
        if isinstance(payload, dict):
            last_state_ns = time.monotonic_ns()
            latest_state.clear()
            latest_state.update(payload)
            write("state", payload)

    def display():
        if last_state_ns is None:
            feedback = "等待 DH5 节点反馈"
        else:
            age_s = (time.monotonic_ns() - last_state_ns) / 1e9
            feedback = f"DH5 反馈已中断 {age_s:.1f}s"
            if age_s < 1.0:
                feedback = (
                    f"节点接收序号={latest_state.get('received_seq')} "
                    f"收到轴={latest_state.get('received_axis')} "
                    f"寄存器方向轴={latest_state.get('applied_axis')} "
                    f"目标={latest_state.get('target_position')} 实测={latest_state.get('actual_position')} "
                    f"状态={latest_state.get('grip_status')} 速度档={latest_state.get('velocity_setting')} "
                    f"控制={latest_state.get('control_state')} "
                    f"{latest_state.get('command_error') or ''} {latest_state.get('feedback_error') or ''}"
                )
        print(
            f"seq={latest_vr.get('pose_seq')} "
            f"X/Y={latest_vr.get('right_stick_x')}/{latest_vr.get('right_stick_y')} "
            f"输入轴={latest_vr.get('gripper_velocity_axis')} "
            f"{feedback}",
            flush=True,
        )

    def receive_json(callback):
        def receive(message):
            try:
                callback(message)
            except (ValueError, TypeError):
                node.get_logger().warning("Invalid monitor payload", throttle_duration_sec=2.0)
        return receive

    node.create_subscription(String, topic, receive_json(forward), 1)
    node.create_subscription(String, "/dh5/state", receive_json(receive_state), 10)
    node.create_subscription(
        UInt16, "/dh5/target_position", lambda msg: write("target", {"target_position": msg.data}), 20
    )
    node.create_timer(0.2, display)


if __name__ == "__main__":
    main()
