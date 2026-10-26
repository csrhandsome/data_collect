#!/usr/bin/env bash
# Two D435 color streams. Topics match config/train/panda.yaml:
#   /camera/external/color/image_raw  (serial 825412070292)
#   /camera/wrist/color/image_raw     (serial 825412070487)
set -euo pipefail

launch_one() {
  local name="$1"
  local serial="$2"
  ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=camera \
    camera_name:="${name}" \
    serial_no:="'${serial}'" \
    enable_depth:=false \
    enable_infra:=false \
    enable_infra1:=false \
    enable_infra2:=false \
    'rgb_camera.color_profile:=640,480,30' \
    rgb_camera.color_format:=RGB8 \
    pointcloud.enable:=false \
    align_depth.enable:=false \
    publish_tf:=false
}

launch_one external 825412070292 &
external_pid=$!
launch_one wrist 825412070487 &
wrist_pid=$!
trap 'kill "$external_pid" "$wrist_pid" 2>/dev/null || true' INT TERM
wait "$external_pid" "$wrist_pid"
