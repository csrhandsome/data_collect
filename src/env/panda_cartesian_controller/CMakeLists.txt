cmake_minimum_required(VERSION 3.22)
project(panda_cartesian_controller)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(franka_semantic_components REQUIRED)
find_package(franka_hardware REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(Franka REQUIRED)

add_library(${PROJECT_NAME} SHARED src/cartesian_pose_impedance_controller.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC
  Franka::Franka
  controller_interface::controller_interface
  hardware_interface::hardware_interface
  pluginlib::pluginlib
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  franka_semantic_components::franka_semantic_components
  realtime_tools::realtime_tools
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
)
if(${hardware_interface_VERSION} VERSION_GREATER_EQUAL "4.20")
  target_compile_definitions(${PROJECT_NAME} PRIVATE HW_HAS_SET_NODISCARD)
endif()
if(${hardware_interface_VERSION} VERSION_GREATER_EQUAL "4.27")
  target_compile_definitions(${PROJECT_NAME} PRIVATE HW_HAS_GET_OPT)
endif()

pluginlib_export_plugin_description_file(controller_interface panda_cartesian_controller.xml)

install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
