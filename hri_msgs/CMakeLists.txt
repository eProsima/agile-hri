cmake_minimum_required(VERSION 3.8)
project(hri_msgs)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ImageDescription.action"
  "action/Stt.action"
  "action/Tts.action"

  "msg/Audio.msg"
  "msg/Expression.idl"
  "msg/Face2D.idl"
  "msg/Face2DList.msg"
  "msg/FaceInterface.msg"
  "msg/FacialLandmarks.idl"
  "msg/Intent.msg"
  "msg/NormalizedPointOfInterest2D.msg"
  "msg/NormalizedRegionOfInterest2D.idl"
  "msg/Pose3D.idl"
  "msg/Skeleton2D.idl"
  "msg/Skeleton2DList.msg"
  "msg/Skeleton3D.idl"
  "msg/Skeleton3DList.msg"

  "srv/PersonID.srv"
  "srv/Reload.srv"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
