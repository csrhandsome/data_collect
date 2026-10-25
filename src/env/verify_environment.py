"""Check Python requirements, ROS message bindings and official controller linking.

Run inside the built container. This opens no robot/FCI connections.
"""

import ctypes
import importlib
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


def main():
    if sys.version_info[:2] != (3, 12) or os.environ.get("ROS_DISTRO") != "jazzy":
        raise RuntimeError("Expected system Python 3.12 and sourced ROS 2 Jazzy")
    subprocess.run(["uv", "pip", "check", "--python", sys.executable], check=True)
    for name in (
        "rclpy",
        "franka_msgs.msg",
        "geometry_msgs.msg",
        "sensor_msgs.msg",
        "numpy",
        "scipy",
        "pyarrow",
        "av",
        "cv2",
        "torch",
        "lerobot.datasets.lerobot_dataset",
        "teleop_xr",
        "pyrealsense2",
        "sounddevice",
        "fastapi",
        "msgpack",
        "control.vr_input",
        "control.collection.pipeline",
        "control.inference.loop",
        "control.base_runner",
        "control.runners",
        "control.direct.franka_runner_stable",
        "control.ros2.franka_runner",
        "control.ros2.camera_connector",
        "control.ros2.microphone_connector",
        "replay.backend.main",
    ):
        importlib.import_module(name)
        print(f"Import OK: {name}", flush=True)

    plugin = Path(
        "/opt/franka_ws/install/franka_example_controllers/lib/libfranka_example_controllers.so"
    )
    ctypes.CDLL(str(plugin))
    print(f"Official C++ controller linking OK: {plugin}", flush=True)

    import rclpy
    from franka_msgs.msg import FrankaRobotState
    from geometry_msgs.msg import PoseStamped
    from rclpy.context import Context
    from rclpy.signals import SignalHandlerOptions

    context = Context()
    rclpy.init(args=[], context=context, signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("data_collect_environment_check", context=context)
    try:
        # Load generated type support in the same interpreter as the Python deps.
        node.create_publisher(PoseStamped, "~/test_pose", 1)
        node.create_subscription(FrankaRobotState, "~/test_state", lambda message: None, 1)
    finally:
        node.destroy_node()
        context.try_shutdown()
        context.destroy()
    print("ROS PoseStamped/FrankaRobotState type support OK", flush=True)
    for name in ("numpy", "torch", "lerobot", "pyarrow", "teleop-xr"):
        print(f"{name}=={version(name)}", flush=True)
    print("Dependency checks passed; robot/controller behavior still requires hardware validation.")


if __name__ == "__main__":
    main()
