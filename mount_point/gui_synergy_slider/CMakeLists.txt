cmake_minimum_required(VERSION 2.8.3)
project(gui_synergy_slider)

find_package(catkin REQUIRED COMPONENTS
  controller_manager_msgs
  rospy
  rqt_gui
  rqt_gui_py
  sr_hand
  sr_robot_msgs
  std_msgs
)

catkin_package(
  DEPENDS
  CATKIN_DEPENDS rospy rqt_gui rqt_gui_py controller_manager_msgs sr_robot_msgs std_msgs sr_hand
  INCLUDE_DIRS
  LIBRARIES
)
catkin_python_setup()

include_directories(
  ${catkin_include_dirs}
)

install( DIRECTORY uis DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION} )

install( FILES gui_synergy_slider_plugin.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION} )

