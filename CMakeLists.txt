cmake_minimum_required(VERSION 2.8.3)
project(ros_rgbd_cnn)

set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(BUILD_CPP_BINDINGS TRUE)

if (BUILD_CPP_BINDINGS)
    add_subdirectory(torch_cpp)
endif()

find_package(catkin REQUIRED COMPONENTS std_msgs sensor_msgs message_generation)

catkin_python_setup()

add_message_files(
  FILES
    Result.msg
)


generate_messages(
  DEPENDENCIES std_msgs sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS message_runtime
  LIBRARIES ${PROJECT_NAME}
)

set(LIBTORCH_LOCAL_FOLDER /home/arwillis/bin/libtorch)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.0)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_PREFIX_PATH "${LIBTORCH_LOCAL_FOLDER}")
#set(CMAKE_MODULE_PATH "${LIBTORCH_LOCAL_FOLDER}/libtorch/share/cmake/")
#set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
#list(APPEND TORCH_LIBRARIES libtorch/lib)
#set(TORCH_LIBRARIES torch)
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} torch_cpp/planenet.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)


#############
## Install ##
#############

install(PROGRAMS
  nodes/ros_rgbd_cnn_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  msg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


#############
## Testing ##
#############

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
