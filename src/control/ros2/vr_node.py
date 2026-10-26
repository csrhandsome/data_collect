"""Publish Teleop XR state for the ROS collection process.

Run this in its own process. It uses the in-process reader and does not start
the multiprocessing VR worker that panda-py collection uses.
"""

import os
import threading
from pathlib import Path

from control.config import DEFAULT_CONFIG, load_config
from control.ros2.vr_connector import dumps_vr


def main(argv=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    from control.collection.controller_hints import build_controller_hints
    from control.vr_input import VRInputReader

    config = load_config(Path(os.environ.get("DATA_COLLECT_CONFIG", DEFAULT_CONFIG)))
    cfg = config.get("vr", {})
    topic = cfg.get("ros2", {}).get("topic", "/vr/input")
    reader = VRInputReader(
        host=cfg.get("host", "0.0.0.0"),
        port=int(cfg.get("port", 4443)),
        long_press_s=float(cfg.get("long_press_s", 0.5)),
        position_alpha=float(cfg.get("position_alpha", 0.6)),
        rotation_alpha=float(cfg.get("rotation_alpha", 0.35)),
        controller_hints=build_controller_hints(config),
    )
    threading.Thread(target=reader.run, name="teleop-xr", daemon=True).start()

    rclpy.init(args=argv)
    node = Node("vr_input_publisher")
    publisher = node.create_publisher(
        String, topic, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
    )

    def publish():
        publisher.publish(String(data=dumps_vr(reader.latest)))

    node.create_timer(0.01, publish)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
