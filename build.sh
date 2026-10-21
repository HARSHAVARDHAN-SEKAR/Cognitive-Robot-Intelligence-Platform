#!/usr/bin/env bash
# Build the CRIP workspace natively (requires ROS 2 Jazzy sourced).
set -e
cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
echo
echo "Build complete. Now run:  source install/setup.bash"
