cmake_minimum_required(VERSION 3.5)
project(ckmeans_1d_dp_stil VERSION 0.0.1)
find_package(pybind11 CONFIG)

set(CMAKE_CXX_STANDARD 20)

set(SOURCE_DIR src)
set(TEST_DIR test)

add_subdirectory(${SOURCE_DIR}/stil_internship)
add_subdirectory(${TEST_DIR}/cpp)

