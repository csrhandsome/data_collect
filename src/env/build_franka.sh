#!/usr/bin/env bash
# Panda / FER driver for robot server protocol 5.
# Official franka_ros2 Jazzy with libfranka >= 0.15 speaks a newer FCI protocol
# and does not match this Panda. These pins are the mul-cps FER1 port, which
# CI-builds on ROS 2 Jazzy, and its libfranka fork: the 0.9.2 protocol plus the
# Ubuntu 24.04 build fixes. Do not replace them with official 0.20.x.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
: "${FRANKA_ROS2_REF:?FRANKA_ROS2_REF is required}"
: "${LIBFRANKA_REF:?LIBFRANKA_REF is required}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
export MAKEFLAGS="${MAKEFLAGS:--j${CMAKE_BUILD_PARALLEL_LEVEL}}"
mkdir -p /opt/franka_ws/src
cd /opt/franka_ws
rm -rf src/franka_ros2 src/libfranka src/panda_cartesian_controller
git clone https://github.com/mul-cps/franka_ros2.git src/franka_ros2
git -C src/franka_ros2 checkout "${FRANKA_ROS2_REF}"
git clone --recursive https://github.com/mul-cps/libfranka.git src/libfranka
git -C src/libfranka checkout "${LIBFRANKA_REF}"
git -C src/libfranka submodule update --init --recursive
cp -a /tmp/panda_cartesian_controller src/panda_cartesian_controller
packages=(
  libfranka
  franka_description
  franka_msgs
  franka_hardware
  franka_semantic_components
  franka_gripper
  franka_robot_state_broadcaster
  panda_cartesian_controller
)
paths=(
  src/libfranka
  src/panda_cartesian_controller
  src/franka_ros2/franka_description
  src/franka_ros2/franka_msgs
  src/franka_ros2/franka_hardware
  src/franka_ros2/franka_semantic_components
  src/franka_ros2/franka_gripper
  src/franka_ros2/franka_robot_state_broadcaster
)
colcon build --paths "${paths[@]}" --packages-select "${packages[@]}" --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
git -C src/franka_ros2 rev-parse HEAD > /opt/franka_ws/FRANKA_ROS2_COMMIT
git -C src/libfranka rev-parse HEAD > /opt/franka_ws/LIBFRANKA_COMMIT
test -f /opt/franka_ws/install/panda_cartesian_controller/lib/libpanda_cartesian_controller.so
