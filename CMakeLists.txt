cmake_minimum_required(VERSION 2.8)
project(cuda_test)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
find_package(OpenCV REQUIRED) 
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA)

CUDA_ADD_EXECUTABLE(g ripple.1.cu)
target_link_libraries(g ${OpenCV_LIBS})

add_executable(c ripple.2.cpp)
target_link_libraries(c ${OpenCV_LIBS})
