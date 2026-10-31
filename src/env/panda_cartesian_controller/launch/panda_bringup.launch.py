"""Panda bringup: FER driver, state broadcaster, external Cartesian impedance."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def load_robot_config(path):
    values = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.endswith(":") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def spawn(context):
    from control.ros2.controller_config import write_controller_config
    from control.yaml_config import load_yaml_config

    config_path = LaunchConfiguration("robot_config_file").perform(context)
    robot = load_robot_config(config_path)
    if robot.get("robot_type") != "panda":
        raise RuntimeError(f"{config_path} must set robot_type panda for this driver")
    robot_ip = robot["robot_ip"]
    load_gripper = robot.get("load_gripper", "true")
    if robot.get("use_fake_hardware", "false").lower() == "true":
        raise RuntimeError("This Panda launch does not start fake hardware")
    share = Path(get_package_share_directory("panda_cartesian_controller"))
    controllers = write_controller_config(share / "config" / "controllers.yaml", load_yaml_config())
    xacro_file = str(
        Path(get_package_share_directory("franka_description")) / "robots" / "panda_arm.urdf.xacro"
    )
    robot_description = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            xacro_file,
            " arm_id:=panda hand:=",
            load_gripper,
            " robot_ip:=",
            robot_ip,
            " use_fake_hardware:=false fake_sensor_commands:=false",
        ]
    )
    description = {"robot_description": robot_description}
    actions = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[description],
        ),
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            parameters=[
                {"source_list": ["franka/joint_states", "panda_gripper/joint_states"], "rate": 30}
            ],
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[description, controllers],
            output="screen",
            on_exit=Shutdown(),
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "--param-file",
                controllers,
                "--controller-ros-args",
                "--remap joint_states:=/franka/joint_states",
                "joint_state_broadcaster",
            ],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["--param-file", controllers, "franka_robot_state_broadcaster"],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["--param-file", controllers, "cartesian_pose_impedance_controller"],
            output="screen",
        ),
    ]
    if load_gripper.lower() == "true":
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("franka_gripper"), "launch", "gripper.launch.py"]
                    )
                ),
                launch_arguments={
                    "robot_ip": robot_ip,
                    "use_fake_hardware": "false",
                    "arm_id": "panda",
                }.items(),
            )
        )
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_config_file",
                default_value="/workspace/src/env/franka.config.yaml",
            ),
            OpaqueFunction(function=spawn),
        ]
    )
