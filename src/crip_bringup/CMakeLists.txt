cmake_minimum_required(VERSION 3.8)
project(crip_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config worlds rviz scripts
  DESTINATION share/${PROJECT_NAME}
)

# status_monitor.py: run it directly for a readable live view of the robot
install(
  PROGRAMS scripts/status_monitor.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
