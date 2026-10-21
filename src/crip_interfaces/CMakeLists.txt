cmake_minimum_required(VERSION 3.8)
project(crip_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SemanticEntity.msg"
  "msg/Task.msg"
  "msg/SceneAssessment.msg"
  "srv/QueryEntities.srv"
  "srv/HaveISeen.srv"
  "srv/PlanMission.srv"
  "srv/AnalyzeScene.srv"
  "srv/RecordExperience.srv"
  "srv/RecallExperience.srv"
  "srv/ExplainDecision.srv"
  DEPENDENCIES geometry_msgs builtin_interfaces
)

ament_package()
