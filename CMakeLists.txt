########################################
# Borrowed and adapted from TVM project
########################################
cmake_minimum_required(VERSION 2.8)
project(dgl C CXX)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# include directories
include_directories("include")
include_directories("third_party/dlpack/include")
include_directories("third_party/dmlc-core/include")

# initial variables
set(DGL_LINKER_LIBS "")
if(MSVC OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
set(DGL_RUNTIME_LINKER_LIBS "")
else(MSVC OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
set(DGL_RUNTIME_LINKER_LIBS "rt")
endif(MSVC OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")

# Generic compilation options
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-DNOMINMAX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
  if(USE_MSVC_MT)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
else(MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
  set(CMAKE_C_FLAGS "-O2 -Wall -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")

  include(FindOpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
  endif(OPENMP_FOUND)
endif(MSVC)

# Source file lists
file(GLOB CORE_SRCS src/graph/*.cc src/graph/network/* src/*.cc src/scheduler/*.cc)

file(GLOB RUNTIME_SRCS src/runtime/*.cc)


add_library(dgl SHARED ${CORE_SRCS} ${RUNTIME_SRCS})

target_link_libraries(dgl ${DGL_LINKER_LIBS} ${DGL_RUNTIME_LINKER_LIBS})

# Installation rules
install(TARGETS dgl DESTINATION lib${LIB_SUFFIX})
