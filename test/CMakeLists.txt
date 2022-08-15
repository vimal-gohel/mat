cmake_minimum_required (VERSION 3.20)

# Project name
project(mat_test)

find_package (GTest REQUIRED)

include_directories (${CMAKE_INSTALL_PREFIX}/include ${GTEST_INCLUDE_DIRS})

add_executable (mat_test mat_operations.cpp)

target_link_libraries(mat_test PRIVATE ${GTEST_LIBRARIES} pthread mat)

install (TARGETS mat_test RUNTIME DESTINATION bin/test)
