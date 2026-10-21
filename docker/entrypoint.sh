#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash
if [ -f /crip_ws/install/setup.bash ]; then
  source /crip_ws/install/setup.bash
fi
exec "$@"
