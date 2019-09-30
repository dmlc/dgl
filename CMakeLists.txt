cmake_minimum_required(VERSION 3.5)
########################################
# Borrowed and adapted from TVM project
########################################
project(dgl C CXX)
message(STATUS "Start configuring project ${PROJECT_NAME}")

# cmake utils
include(cmake/util/Util.cmake)
include(cmake/util/MshadowUtil.cmake)
include(cmake/util/FindCUDA.cmake)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alernatively, use cmake -DOPTION=VALUE through command-line.
dgl_option(USE_CUDA "Build with CUDA" OFF)
dgl_option(USE_OPENMP "Build with OpenMP" ON)
dgl_option(BUILD_CPP_TEST "Build cpp unittest executables" OFF)

if(USE_CUDA)
  message(STATUS "Build with CUDA support")
  project(dgl C CXX)
  include(cmake/modules/CUDA.cmake)
endif(USE_CUDA)

# include directories
include_directories("include")
include_directories("third_party/dlpack/include")
include_directories("third_party/dmlc-core/include")
include_directories("third_party/minigun/minigun")
include_directories("third_party/minigun/third_party/moderngpu/src")

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
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
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
endif(MSVC)

if(USE_OPENMP)
  include(FindOpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
  endif(OPENMP_FOUND)
endif(USE_OPENMP)

# configure minigun
add_definitions(-DENABLE_PARTIAL_FRONTIER=0)  # disable minigun partial frontier compile

# Source file lists
file(GLOB DGL_SRC
  src/*.cc
  src/array/*.cc
  src/array/cpu/*.cc
  src/kernel/*.cc
  src/kernel/cpu/*.cc
  src/runtime/*.cc
)

file(GLOB_RECURSE DGL_SRC_1
  src/api/*.cc
  src/graph/*.cc
  src/scheduler/*.cc
)

list(APPEND DGL_SRC ${DGL_SRC_1})

# Configure cuda
if(USE_CUDA)
  dgl_config_cuda(DGL_CUDA_SRC)
  list(APPEND DGL_SRC ${DGL_CUDA_SRC})
endif(USE_CUDA)

if(USE_CUDA)
  cuda_add_library(dgl SHARED ${DGL_SRC})
else(USE_CUDA)
  add_library(dgl SHARED ${DGL_SRC})
endif(USE_CUDA)

# For serialization
add_subdirectory("third_party/dmlc-core")
list(APPEND DGL_LINKER_LIBS dmlc)
set(GOOGLE_TEST 0) # Turn off dmlc-core test

target_link_libraries(dgl ${DGL_LINKER_LIBS} ${DGL_RUNTIME_LINKER_LIBS})

# Installation rules
install(TARGETS dgl DESTINATION lib${LIB_SUFFIX})

# Testing
if(BUILD_CPP_TEST)
  message(STATUS "Build with unittest")
  add_subdirectory(./third_party/googletest)
  enable_testing()
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  file(GLOB_RECURSE TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/tests/cpp/*.cc)
  add_executable(runUnitTests ${TEST_SRC_FILES})
  target_link_libraries(runUnitTests gtest gtest_main)
  target_link_libraries(runUnitTests dgl)
  add_test(UnitTests runUnitTests)
endif(BUILD_CPP_TEST)
