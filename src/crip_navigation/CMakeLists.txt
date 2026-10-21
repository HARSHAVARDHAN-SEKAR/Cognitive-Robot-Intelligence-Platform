cmake_minimum_required(VERSION 3.8)
project(crip_navigation)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config maps launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
