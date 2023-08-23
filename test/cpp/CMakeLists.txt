include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(KMEANS_1D_NLOGU_ROOT_DIR ${SOURCE_DIR}/algo/codec/kmeans_1d_nlogu_c)
set(KMEANS_1D_NLOGU_INCLUDE_DIR ${KMEANS_1D_NLOGU_ROOT_DIR}/include)
set(KMEANS_1D_NLOGU_SRC_DIR ${KMEANS_1D_NLOGU_ROOT_DIR}/src)
set(TEST_INCLUDE_DIR ${INCLUDE_DIR} ${KMEANS_1D_NLOGU_INCLUDE_DIR})

add_executable(
  stil_internship_test	
  "test_metrics.cpp"
  ${KMEANS_1D_NLOGU_INCLUDE_DIR}/KMeans.h
  ${KMEANS_1D_NLOGU_INCLUDE_DIR}/metrics.h
  ${KMEANS_1D_NLOGU_INCLUDE_DIR}/SMAWK.h
  ${KMEANS_1D_NLOGU_SRC_DIR}/metrics.cpp
  ${KMEANS_1D_NLOGU_SRC_DIR}/KMeans.cpp
  ${KMEANS_1D_NLOGU_SRC_DIR}/SMAWK.cpp
 "test_KMeans.cpp" "test_SMAWK.cpp")
add_compile_definitions(TEST_SUITE_COMPILATION)
target_include_directories(stil_internship_test PRIVATE ${TEST_INCLUDE_DIR})
target_link_libraries(
  stil_internship_test
  _kmeans_nlogu
  GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(stil_internship_test)
