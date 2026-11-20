cmake_minimum_required(VERSION 3.8)
project(omnigda_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Object3D.msg"
  "msg/Object3DArray.msg"
  "msg/GraspPose.msg"
  "msg/GraspPoseArray.msg"
  "msg/SceneRelation.msg"
  "msg/SceneGraph.msg"
  "msg/PerceptionMetrics.msg"
  "srv/RegisterObject.srv"
  "srv/ListObjects.srv"
  "srv/QueryScene.srv"
  "action/ExecuteGrasp.action"
  "action/ExecuteInstruction.action"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
