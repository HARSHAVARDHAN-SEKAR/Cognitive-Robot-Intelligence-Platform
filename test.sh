#!/usr/bin/env bash
# Run the CRIP simulation/integration tests.
set -e
cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
colcon test --packages-select crip_tests --event-handlers console_direct+
colcon test-result --verbose
