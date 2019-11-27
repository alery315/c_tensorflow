cmake_minimum_required(VERSION 3.14)
project(c_tensorflow C)

set(CMAKE_C_STANDARD 99)

set(INC_DIR /home/alery/libtensorflow/include)
set(LINK_DIR /home/alery/libtensorflow/lib)

include_directories(${INC_DIR})
link_directories(${LINK_DIR})
link_libraries(tensorflow)

add_executable(c_tensorflow use_pb.c)

target_link_libraries(c_tensorflow tensorflow)

