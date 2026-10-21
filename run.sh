#!/usr/bin/env bash
# Launch the full autonomy demo (Gazebo + RViz + perception + brain).
set -e
cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch crip_bringup autonomy.launch.py "$@"
