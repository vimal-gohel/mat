cmake_minimum_required (VERSION 3.20)

project (mat)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/output)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# C++ Standard
# Default to C++14
if (NOT CMAKE_CXX_STANDARD)
  set (CMAKE_CXX_STANDARD 14)
endif ()
set (CMAKE_CXX_STANDARD_REQUIRED TRUE)
set (CMAKE_CXX_EXTENSIONS OFF)

# Enable Debug support
set (CMAKE_BUILD_TYPE Debug)

if (WIN32)
  # Add __declspec(dllimport) for all global symbols
  set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

add_subdirectory(code)
#add_subdirectory(test)
add_subdirectory(examples)
