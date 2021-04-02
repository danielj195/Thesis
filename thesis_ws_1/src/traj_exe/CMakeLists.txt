cmake_minimum_required(VERSION 2.8.12)
project(traj_exe)
add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
  roscpp 
  rospy
  abb_libegm
  genmsg
  actionlib_msgs
  std_msgs
  sensor_msgs
  geometry_msgs
#   arm_sanding_utilities
)

###################################
## catkin specific configuration ##
###################################

#[[
add_message_files(
   FILES
 )
]]

add_service_files(
  FILES
#   JointStates.srv
  ForceReading.srv
  LaserReading.srv
)


generate_messages(
  DEPENDENCIES
  std_msgs
  ##sensor_msgs
  ##actionlib_msgs
  ##geometry_msgs
)

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS 
    roscpp 
    abb_libegm 
    system_lib 
    message_runtime 
    tf 
    actionlib_msgs
)

###########
## Build ##
###########


include_directories(include
                    ${PYTHON_INCLUDE_DIRS}
                    ${catkin_INCLUDE_DIRS}
                    )





####################################
# Build the advanced level samples #
####################################


# add_executable(connect_node src/connect_node.cpp src/connect.cpp)
# target_link_libraries(connect_node ${catkin_LIBRARIES} ${abb_libegm_LIBRARIES})


