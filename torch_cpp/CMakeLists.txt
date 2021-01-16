cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch_cpp_cnn_inference)

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
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(libtorch_cpp_cnn_inference example-app.cpp)
#link_directories(/usr/lib/x86_64-linux-gnu)
#link_directories(${LIBTORCH_LOCAL_FOLDER}/lib)
#set(CUDA_cublas_LIBRARY /usr/lib/x86_64-linux-gnu/libcublas.so)
target_link_libraries(libtorch_cpp_cnn_inference "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}")
set_property(TARGET libtorch_cpp_cnn_inference PROPERTY CXX_STANDARD 14)
