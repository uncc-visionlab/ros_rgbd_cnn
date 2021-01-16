cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(planenet_cpp)

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
include_directories(SYSTEM ../include ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_library(planenet SHARED planenet.cpp)
target_link_libraries(planenet "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}")
set_property(TARGET planenet PROPERTY CXX_STANDARD 14)

add_executable(planenet_demo planenet_demo.cpp)
target_link_libraries(planenet_demo planenet "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}")
set_property(TARGET planenet_demo PROPERTY CXX_STANDARD 14)
