cmake_minimum_required(VERSION 3.16)
project(arm_sim_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# RoboStack on macOS: ROS targets pull in lib<pkg>__rosidl_generator_py, which
# aborts a pure C++ binary at load. Dropping unused dylibs avoids it.
# See https://github.com/RoboStack/ros-rolling/pull/51
if(APPLE)
  add_link_options("LINKER:-dead_strip_dylibs")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(fake_arm_node src/fake_arm_node.cpp)
# Specific targets, not ament_target_dependencies(): see the linker note above.
target_link_libraries(fake_arm_node PUBLIC
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs__rosidl_typesupport_cpp)

install(TARGETS fake_arm_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
