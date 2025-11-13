# Copyright 2025 Proyectos y Sistemas de Mantenimiento SL (eProsima).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)
project(vulcanexus_hri_cpp)

# Global flags
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(OpenCV REQUIRED)
find_package(magic_enum CONFIG REQUIRED)

# Targets
set(LIBRARY_NAME "hri")
ament_auto_add_library(${LIBRARY_NAME} SHARED DIRECTORY src/hri)
target_link_libraries(${LIBRARY_NAME} magic_enum::magic_enum)

option(COMPILE_EXAMPLE "Build example" ON)
if (COMPILE_EXAMPLE)
    find_package(cv_bridge REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(sensor_msgs REQUIRED)

    add_executable(${PROJECT_NAME}_example_faces src/example_faces.cpp)
    ament_target_dependencies(${PROJECT_NAME}_example_faces
        rclcpp
        sensor_msgs
        cv_bridge
    )
    target_link_libraries(${PROJECT_NAME}_example_faces ${LIBRARY_NAME} ${OpenCV_LIBRARIES})

    add_executable(${PROJECT_NAME}_example_bodies src/example_bodies.cpp)
    ament_target_dependencies(${PROJECT_NAME}_example_bodies
        rclcpp
        sensor_msgs
        cv_bridge
    )
    target_link_libraries(${PROJECT_NAME}_example_bodies ${LIBRARY_NAME} ${OpenCV_LIBRARIES})

    add_executable(${PROJECT_NAME}_example_voice src/example_voices.cpp)
    ament_target_dependencies(${PROJECT_NAME}_example_voice
        rclcpp
    )
    target_link_libraries(${PROJECT_NAME}_example_voice ${LIBRARY_NAME})
endif()

# Tests
if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()

    find_package(ament_cmake_gtest REQUIRED)
    add_subdirectory(test)
endif ()

ament_auto_package()
