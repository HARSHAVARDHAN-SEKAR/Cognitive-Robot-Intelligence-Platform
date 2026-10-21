cmake_minimum_required(VERSION 3.8)
project(crip_tests)

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/test_pipeline.py)
endif()

ament_package()
