cmake_minimum_required(VERSION 3.20)

project(matrix_2x2)

add_executable(${PROJECT_NAME}
  matrix_2x2.cpp
)

include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_INSTALL_PREFIX}/lib/libmat.so)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
