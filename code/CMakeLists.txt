cmake_minimum_required (VERSION 3.20)

# Project Name
#project (mat)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

include_directories (include)

add_library (${PROJECT_NAME} SHARED src/mat.cpp)

install (TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
install (FILES include/mat.hpp DESTINATION include)
