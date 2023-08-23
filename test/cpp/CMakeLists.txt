include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  stil_internship_test
  test_tests.cpp
)

target_link_libraries(
  stil_internship_test
  GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(stil_internship_test)
