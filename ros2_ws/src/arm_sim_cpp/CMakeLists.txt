cmake_minimum_required(VERSION 3.16)
project(arm_sim_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# macOS + RoboStack: rclcpp's and every <pkg>_TARGETS list export
# lib<pkg>__rosidl_generator_py (the Python bindings), which reference CPython
# symbols and abort a pure C++ executable at load:
#   dyld: symbol not found in flat namespace '_PyExc_RuntimeError'
# conda-build links RoboStack's own binaries with -dead_strip_dylibs, which drops
# any dylib no symbol is used from. Do the same here so we match.
# Upstream fix (Rolling only so far): https://github.com/RoboStack/ros-rolling/pull/51
if(APPLE)
  add_link_options("LINKER:-dead_strip_dylibs")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(fake_arm_node src/fake_arm_node.cpp)
# Modern CMake targets. Link the specific C++ typesupport target rather than
# ${sensor_msgs_TARGETS} to keep the link line minimal.
target_link_libraries(fake_arm_node PUBLIC
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs__rosidl_typesupport_cpp)

install(TARGETS fake_arm_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
