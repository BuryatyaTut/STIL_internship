cmake_minimum_required(VERSION 3.5)
project(ckmeans_1d_dp_stil VERSION 0.0.1)
find_package(pybind11 CONFIG)

set(CMAKE_CXX_STANDARD 20)
file(REAL_PATH src/stil_internship SOURCE_DIR)
#set(SOURCE_DIR src/stil_internship)
set(TEST_DIR test)

add_subdirectory(${SOURCE_DIR})
add_subdirectory(${TEST_DIR}/cpp)

