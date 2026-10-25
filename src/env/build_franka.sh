#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export MAKEFLAGS="${MAKEFLAGS:--j2}"
mkdir -p /opt/franka_ws/src
cd /opt/franka_ws
git clone --filter=blob:none https://github.com/frankarobotics/franka_ros2.git src/franka_ros2
git -C src/franka_ros2 checkout "${FRANKA_ROS2_REF}"
git clone --branch 2.9.0 --depth 1 https://github.com/frankarobotics/franka_description.git src/franka_description
git clone --recursive --branch 0.20.5 --depth 1 https://github.com/frankarobotics/libfranka.git src/libfranka
packages=(libfranka franka_description franka_msgs franka_hardware franka_semantic_components
          franka_gripper franka_robot_state_broadcaster franka_example_controllers)
# Discover only this single-arm stack. The upstream repo also contains optional
# mobile/Gazebo packages referenced by bringup; discovering those would require
# building them even though example.launch.py does not use them.
paths=(src/libfranka src/franka_description src/franka_ros2/franka_bringup)
for package in "${packages[@]:2}"; do paths+=("src/franka_ros2/$package"); done
colcon build --paths "${paths[@]}" --packages-select "${packages[@]}" franka_bringup --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
git -C src/franka_ros2 rev-parse HEAD > /opt/franka_ws/FRANKA_ROS2_COMMIT
