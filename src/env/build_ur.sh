#!/usr/bin/env bash
# Build the official Jazzy driver in a separate overlay; Franka stays installed.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
: "${UR_ROS2_REF:?UR_ROS2_REF is required}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
export MAKEFLAGS="${MAKEFLAGS:--j${CMAKE_BUILD_PARALLEL_LEVEL}}"
mkdir -p /opt/ur_ws/src
cd /opt/ur_ws
git clone https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git src/ur_driver
git -C src/ur_driver checkout "$UR_ROS2_REF"
colcon build --base-paths src/ur_driver --packages-up-to ur_robot_driver \
  --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
git -C src/ur_driver rev-parse HEAD > /opt/ur_ws/UR_ROS2_COMMIT
test -f install/ur_robot_driver/lib/libur_robot_driver_plugin.so
