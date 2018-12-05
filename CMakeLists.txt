########################################
# Borrowed and adapted from TVM project
########################################
cmake_minimum_required(VERSION 2.8)
project(dgl C CXX)

# Utility functions
include(cmake/util/Util.cmake)

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
#tvm_option(USE_CUDA "Build with CUDA" OFF)
#tvm_option(USE_OPENCL "Build with OpenCL" OFF)
#tvm_option(USE_VULKAN "Build with Vulkan" OFF)
#tvm_option(USE_OPENGL "Build with OpenGL" OFF)
#tvm_option(USE_METAL "Build with Metal" OFF)
#tvm_option(USE_ROCM "Build with ROCM" OFF)
#tvm_option(ROCM_PATH "The path to rocm" /opt/rocm)
#tvm_option(USE_RPC "Build with RPC" ON)
#tvm_option(USE_LLVM "Build with LLVM, can be set to specific llvm-config path" OFF)
#tvm_option(USE_STACKVM_RUNTIME "Include stackvm into the runtime" OFF)
#tvm_option(USE_GRAPH_RUNTIME "Build with tiny graph runtime" ON)
#tvm_option(USE_GRAPH_RUNTIME_DEBUG "Build with tiny graph runtime debug mode" OFF)
#tvm_option(USE_RTTI "Build with RTTI" ON)
#tvm_option(USE_MSVC_MT "Build with MT" OFF)
#tvm_option(INSTALL_DEV "Install compiler infrastructure" OFF)

# Contrib library options
#tvm_option(USE_BLAS "The blas library to be linked" none)
#tvm_option(USE_MKL_PATH "MKL root path when use MKL blas" none)
#tvm_option(USE_CUDNN "Build with cuDNN" OFF)
#tvm_option(USE_CUBLAS "Build with cuBLAS" OFF)
#tvm_option(USE_MIOPEN "Build with ROCM:MIOpen" OFF)
#tvm_option(USE_ROCBLAS "Build with ROCM:RoCBLAS" OFF)
#tvm_option(USE_SORT "Build with sort support" OFF)
#tvm_option(USE_NNPACK "Build with nnpack support" OFF)
#tvm_option(USE_RANDOM "Build with random support" OFF)

# include directories
include_directories("include")
include_directories("third_party/dlpack/include")
include_directories("third_party/dmlc-core/include")

# initial variables
set(DGL_LINKER_LIBS "")
set(DGL_RUNTIME_LINKER_LIBS "")

# Generic compilation options
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
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
  if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(CMAKE_C_FLAGS "-fopenmp ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fopenmp ${CMAKE_CXX_FLAGS}")
  endif()
endif(MSVC)

# add source group
#FILE(GLOB_RECURSE GROUP_SOURCE "src/*.cc")
#FILE(GLOB_RECURSE GROUP_INCLUDE "src/*.h" "include/*.h")
#assign_source_group("Source" ${GROUP_SOURCE})
#assign_source_group("Include" ${GROUP_INCLUDE})

# Source file lists
file(GLOB CORE_SRCS src/graph/*.cc src/*.cc src/scheduler/*.cc)

file(GLOB RUNTIME_SRCS src/runtime/*.cc)

# Package runtime rules
#if(NOT USE_RTTI)
#  add_definitions(-DDMLC_ENABLE_RTTI=0)
#endif()
#
#if(USE_RPC)
#  message(STATUS "Build with RPC support...")
#  file(GLOB RUNTIME_RPC_SRCS src/runtime/rpc/*.cc)
#  list(APPEND RUNTIME_SRCS ${RUNTIME_RPC_SRCS})
#endif(USE_RPC)
#
#file(GLOB STACKVM_RUNTIME_SRCS src/runtime/stackvm/*.cc)
#file(GLOB STACKVM_CODEGEN_SRCS src/codegen/stackvm/*.cc)
#list(APPEND COMPILER_SRCS ${STACKVM_CODEGEN_SRCS})
#if(USE_STACKVM_RUNTIME)
#  message(STATUS "Build with stackvm support in runtime...")
#  list(APPEND RUNTIME_SRCS ${STACKVM_RUNTIME_SRCS})
#else()
#  list(APPEND COMPILER_SRCS ${STACKVM_RUNTIME_SRCS})
#endif(USE_STACKVM_RUNTIME)
#
#if(USE_GRAPH_RUNTIME)
#  message(STATUS "Build with Graph runtime support...")
#  file(GLOB RUNTIME_GRAPH_SRCS src/runtime/graph/*.cc)
#  list(APPEND RUNTIME_SRCS ${RUNTIME_GRAPH_SRCS})
#
#  if(USE_GRAPH_RUNTIME_DEBUG)
#    set_source_files_properties(${RUNTIME_GRAPH_SRCS}
#      PROPERTIES COMPILE_DEFINITIONS "TVM_GRAPH_RUNTIME_DEBUG")
#  endif(USE_GRAPH_RUNTIME_DEBUG)
#endif(USE_GRAPH_RUNTIME)

# Module rules
#include(cmake/modules/VTA.cmake)
#include(cmake/modules/CUDA.cmake)
#include(cmake/modules/OpenCL.cmake)
#include(cmake/modules/OpenGL.cmake)
#include(cmake/modules/Vulkan.cmake)
#include(cmake/modules/Metal.cmake)
#include(cmake/modules/ROCM.cmake)
#include(cmake/modules/LLVM.cmake)
#include(cmake/modules/contrib/BLAS.cmake)
#include(cmake/modules/contrib/Random.cmake)
#include(cmake/modules/contrib/Sort.cmake)
#include(cmake/modules/contrib/NNPack.cmake)

add_library(dgl SHARED ${CORE_SRCS} ${RUNTIME_SRCS})
#add_library(dgl_runtime SHARED ${RUNTIME_SRCS})

target_link_libraries(dgl ${DGL_LINKER_LIBS} ${DGL_RUNTIME_LINKER_LIBS})
#target_link_libraries(dgl_runtime ${DGL_RUNTIME_LINKER_LIBS})

# Related headers
#target_include_directories(
#  dgl
#  PUBLIC "HalideIR/src"
#  PUBLIC "topi/include")

# Tests
#set(TEST_EXECS "")
#file(GLOB TEST_SRCS tests/cpp/*.cc)
#find_library(GTEST_LIB gtest)

#if(GTEST_LIB)
#  foreach(__srcpath ${TEST_SRCS})
#    get_filename_component(__srcname ${__srcpath} NAME)
#    string(REPLACE ".cc" "" __execname ${__srcname})
#    add_executable(${__execname} ${__srcpath})
#    list(APPEND TEST_EXECS ${__execname})
#    target_link_libraries(${__execname}
#      tvm ${GTEST_LIB} pthread)
#    set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_ALL 1)
#    set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
#  endforeach()
#  add_custom_target(cpptest DEPENDS ${TEST_EXECS})
#endif()

# Custom targets
#add_custom_target(runtime DEPENDS tvm_runtime)

# Installation rules
install(TARGETS dgl DESTINATION lib${LIB_SUFFIX})
#install(TARGETS dgl_runtime DESTINATION lib${LIB_SUFFIX})

#if (INSTALL_DEV)
#  install(
#    DIRECTORY "include/." DESTINATION "include"
#    FILES_MATCHING
#    PATTERN "*.h"
#  )
#  install(
#    DIRECTORY "topi/include/." DESTINATION "include"
#    FILES_MATCHING
#    PATTERN "*.h"
#  )
#  install(
#    DIRECTORY "HalideIR/src/." DESTINATION "include/HalideIR"
#    FILES_MATCHING
#    PATTERN "*.h"
#  )
#  install(
#    DIRECTORY "dlpack/include/." DESTINATION "include"
#    FILES_MATCHING
#    PATTERN "*.h"
#    )
#  install(
#    DIRECTORY "nnvm/include/." DESTINATION "include"
#    FILES_MATCHING
#    PATTERN "*.h"
#    )
#else(INSTALL_DEV)
#  install(
#    DIRECTORY "include/tvm/runtime/." DESTINATION "include/tvm/runtime"
#    FILES_MATCHING
#    PATTERN "*.h"
#    )
#endif(INSTALL_DEV)

# More target definitions
#if(MSVC)
#  target_compile_definitions(tvm PRIVATE -DHalide_EXPORTS)
#  target_compile_definitions(tvm_runtime PRIVATE -DHalide_EXPORTS)
#  target_compile_definitions(tvm PRIVATE -DTVM_EXPORTS)
#  target_compile_definitions(tvm_runtime PRIVATE -DTVM_EXPORTS)
#  target_compile_definitions(nnvm_compiler PRIVATE -DNNVM_EXPORTS)
#endif()
