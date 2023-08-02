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
dgl_option(USE_LIBXSMM "Build with LIBXSMM library optimization" ON)
dgl_option(BUILD_CPP_TEST "Build cpp unittest executables" OFF)
dgl_option(LIBCXX_ENABLE_PARALLEL_ALGORITHMS "Enable the parallel algorithms library. This requires the PSTL to be available." OFF)
dgl_option(USE_S3 "Build with S3 support" OFF)
dgl_option(USE_HDFS "Build with HDFS support" OFF) # Set env HADOOP_HDFS_HOME if needed
dgl_option(REBUILD_LIBXSMM "Clean LIBXSMM build cache at every build" OFF) # Set env HADOOP_HDFS_HOME if needed
dgl_option(USE_EPOLL "Build with epoll for socket communicator" ON)
dgl_option(BUILD_TORCH "Build the PyTorch plugin" OFF)
dgl_option(BUILD_SPARSE "Build DGL sparse library" ON)
dgl_option(BUILD_GRAPHBOLT "Build Graphbolt library" OFF)
dgl_option(TORCH_PYTHON_INTERPS "Python interpreter used to build tensoradapter and DGL sparse library" python3)

# Set debug compile option for gdb, only happens when -DCMAKE_BUILD_TYPE=DEBUG
if (NOT MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -O0 -g3 -ggdb")
endif(NOT MSVC)

if(USE_CUDA)
  message(STATUS "Build with CUDA support")
  project(dgl C CXX)
  # see https://github.com/NVIDIA/thrust/issues/1401
  add_definitions(-DTHRUST_CUB_WRAPPED_NAMESPACE=dgl)
  include(cmake/modules/CUDA.cmake)
  message(STATUS "Use external CUB/Thrust library for a consistent API and performance.")
  cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/thrust")
  cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/thrust/dependencies/cub")
  cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/thrust/dependencies/libcudacxx/include")
endif(USE_CUDA)

# initial variables
if(NOT MSVC)
set(DGL_LINKER_LIBS "dl")
endif(NOT MSVC)

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
  set(CMAKE_C_FLAGS "-O2 -Wall -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
  if(NOT APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--warn-common ${CMAKE_SHARED_LINKER_FLAGS}")
  endif(NOT APPLE)
endif(MSVC)

if(USE_OPENMP)
  include(FindOpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
  endif(OPENMP_FOUND)
  message(STATUS "Build with OpenMP.")
endif(USE_OPENMP)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  message(STATUS "Disabling LIBXSMM on ${CMAKE_SYSTEM_PROCESSOR}.")
  set(USE_LIBXSMM OFF)
endif()

if(USE_LIBXSMM)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_LIBXSMM -DDGL_CPU_LLC_SIZE=40000000")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_LIBXSMM -DDGL_CPU_LLC_SIZE=40000000")
  message(STATUS "Build with LIBXSMM optimization.")
endif(USE_LIBXSMM)

if ((NOT MSVC) AND USE_EPOLL)
  INCLUDE(CheckIncludeFile)
  check_include_file("sys/epoll.h" EPOLL_AVAILABLE)
  if (EPOLL_AVAILABLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_EPOLL")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_EPOLL")
  else()
    message(WARNING "EPOLL is not available on this platform...")
  endif()
endif ()

# To compile METIS correct for DGL.
if(MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DIDXTYPEWIDTH=64 /DREALTYPEWIDTH=32")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DIDXTYPEWIDTH=64 /DREALTYPEWIDTH=32")
else(MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=32")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=32")
endif(MSVC)

# configure minigun
add_definitions(-DENABLE_PARTIAL_FRONTIER=0)  # disable minigun partial frontier compile
# Source file lists
file(GLOB DGL_SRC
  src/*.cc
  src/array/*.cc
  src/array/cpu/*.cc
  src/random/*.cc
  src/random/cpu/*.cc
  src/runtime/*.cc
  src/geometry/*.cc
  src/geometry/cpu/*.cc
  src/partition/*.cc
)

file(GLOB_RECURSE DGL_SRC_1
  src/api/*.cc
  src/graph/*.cc
  src/scheduler/*.cc
)

list(APPEND DGL_SRC ${DGL_SRC_1})

if (NOT MSVC)
  file(GLOB_RECURSE DGL_RPC_SRC src/rpc/*.cc)
else()
  file(GLOB_RECURSE DGL_RPC_SRC src/rpc/network/*.cc)
endif()
list(APPEND DGL_SRC ${DGL_RPC_SRC})

# Configure cuda
if(USE_CUDA)
  dgl_config_cuda(DGL_CUDA_SRC)
  list(APPEND DGL_SRC ${DGL_CUDA_SRC})
  cuda_add_library(dgl SHARED ${DGL_SRC})
else(USE_CUDA)
  add_library(dgl SHARED ${DGL_SRC})
endif(USE_CUDA)

set_property(TARGET dgl PROPERTY CXX_STANDARD 14)

# include directories
target_include_directories(dgl PRIVATE "include")
target_include_directories(dgl PRIVATE "third_party/dlpack/include")
target_include_directories(dgl PRIVATE "third_party/dmlc-core/include")
target_include_directories(dgl PRIVATE "third_party/phmap/")
target_include_directories(dgl PRIVATE "third_party/METIS/include/")
target_include_directories(dgl PRIVATE "tensoradapter/include")
target_include_directories(dgl PRIVATE "third_party/nanoflann/include")
target_include_directories(dgl PRIVATE "third_party/libxsmm/include")
target_include_directories(dgl PRIVATE "third_party/pcg/include")

# For serialization
if (USE_HDFS)
  option(DMLC_HDFS_SHARED "dgl has to build with dynamic hdfs library" ON)
endif()
add_subdirectory("third_party/dmlc-core")
list(APPEND DGL_LINKER_LIBS dmlc)
set(GOOGLE_TEST 0) # Turn off dmlc-core test

# Compile METIS
if(NOT MSVC)
  set(GKLIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/METIS/GKlib")
  include(${GKLIB_PATH}/GKlibSystem.cmake)
  include_directories(${GKLIB_PATH})
  include_directories("third_party/METIS/include/")
  add_subdirectory("third_party/METIS/libmetis/")
  list(APPEND DGL_LINKER_LIBS metis)
endif(NOT MSVC)

# Compile LIBXSMM
if((NOT MSVC) AND USE_LIBXSMM)
  set(LIBXSMM_COMPILER "${CMAKE_C_COMPILER}")
  if (APPLE)
    set(LIBXSMM_COMPILER "${CMAKE_C_COMPILER} -isysroot ${CMAKE_OSX_SYSROOT}")
  endif()
  if(REBUILD_LIBXSMM)
    add_custom_target(libxsmm COMMAND make realclean COMMAND make -j ECFLAGS="-Wno-error=deprecated-declarations" BLAS=0 CC=${LIBXSMM_COMPILER}
                      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/libxsmm
                      )
  else(REBUILD_LIBXSMM)
    add_custom_target(libxsmm COMMAND make -j ECFLAGS="-Wno-error=deprecated-declarations" BLAS=0 CC=${LIBXSMM_COMPILER}
                      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/libxsmm
                      )
  endif(REBUILD_LIBXSMM)
  add_dependencies(dgl libxsmm)
  list(APPEND DGL_LINKER_LIBS -L${CMAKE_SOURCE_DIR}/third_party/libxsmm/lib/ xsmm.a)
endif((NOT MSVC) AND USE_LIBXSMM)

# Avoid exposing third-party symbols when using DGL as a library.
if((NOT MSVC) AND (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--exclude-libs,ALL")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--exclude-libs,ALL")
endif()

# Compile gpu_cache
if(USE_CUDA)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_GPU_CACHE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_GPU_CACHE")
  # Manually build gpu_cache because CMake always builds it as shared
  file(GLOB gpu_cache_src
    third_party/HugeCTR/gpu_cache/src/nv_gpu_cache.cu
  )
  cuda_add_library(gpu_cache STATIC ${gpu_cache_src})
  target_include_directories(gpu_cache PRIVATE "third_party/HugeCTR/gpu_cache/include")
  target_include_directories(dgl PRIVATE "third_party/HugeCTR/gpu_cache/include")
  list(APPEND DGL_LINKER_LIBS gpu_cache)
  message(STATUS "Build with HugeCTR GPU embedding cache.")
endif(USE_CUDA)

# support PARALLEL_ALGORITHMS
if (LIBCXX_ENABLE_PARALLEL_ALGORITHMS)
  add_definitions(-DPARALLEL_ALGORITHMS)
endif(LIBCXX_ENABLE_PARALLEL_ALGORITHMS)

target_link_libraries(dgl ${DGL_LINKER_LIBS} ${DGL_RUNTIME_LINKER_LIBS})
if(MSVC)
  add_custom_command(
    TARGET dgl POST_BUILD COMMAND
    ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:dgl>" "$<TARGET_FILE_DIR:dgl>/..")
endif(MSVC)

# Tensor adapter libraries
# Linking against LibTorch involves linking against a bunch of other libraries
# returned by PyTorch's CMake (e.g. C10 or NVTools).  Because CMake caches
# the found libraries in find_library(), often times CMake will look into the libraries
# of the wrong version when I build everything in the same CMake process.  As
# a result, I (BarclayII) am launching an individual CMake build for every PyTorch version.
if(BUILD_TORCH)
  file(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR} BINDIR)
  file(TO_NATIVE_PATH ${CMAKE_COMMAND} CMAKE_CMD)
  if(MSVC)
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tensoradapter/pytorch/build.bat BUILD_SCRIPT)
    add_custom_target(
      tensoradapter_pytorch
      ${CMAKE_COMMAND} -E env
      CMAKE_COMMAND=${CMAKE_CMD}
      CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
      USE_CUDA=${USE_CUDA}
      BINDIR=${BINDIR}
      cmd /e:on /c ${BUILD_SCRIPT} ${TORCH_PYTHON_INTERPS}
      DEPENDS ${BUILD_SCRIPT}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensoradapter/pytorch)
  else(MSVC)
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tensoradapter/pytorch/build.sh BUILD_SCRIPT)
    add_custom_target(
      tensoradapter_pytorch
      ${CMAKE_COMMAND} -E env
      CMAKE_COMMAND=${CMAKE_CMD}
      CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
      USE_CUDA=${USE_CUDA}
      BINDIR=${CMAKE_CURRENT_BINARY_DIR}
      bash ${BUILD_SCRIPT} ${TORCH_PYTHON_INTERPS}
      DEPENDS ${BUILD_SCRIPT}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensoradapter/pytorch)
  endif(MSVC)
  add_dependencies(dgl tensoradapter_pytorch)
endif(BUILD_TORCH)

# Installation rules
install(TARGETS dgl DESTINATION lib${LIB_SUFFIX})

# Testing
if(BUILD_CPP_TEST)
  message(STATUS "Build with unittest")
  add_subdirectory(./third_party/googletest)
  enable_testing()
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  include_directories("include")
  include_directories("third_party/dlpack/include")
  include_directories("third_party/dmlc-core/include")
  include_directories("third_party/phmap")
  include_directories("third_party/libxsmm/include")
  include_directories("third_party/pcg/include")
  file(GLOB_RECURSE TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/tests/cpp/*.cc)
  add_executable(runUnitTests ${TEST_SRC_FILES})
  target_link_libraries(runUnitTests gtest gtest_main)
  target_link_libraries(runUnitTests dgl)
  add_test(UnitTests runUnitTests)
endif(BUILD_CPP_TEST)

if(BUILD_SPARSE)
  message(STATUS "Configuring DGL sparse library")
  file(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR} BINDIR)
  file(TO_NATIVE_PATH ${CMAKE_COMMAND} CMAKE_CMD)
  get_target_property(DGL_INCLUDE_DIRS dgl INCLUDE_DIRECTORIES)
  message(STATUS "DGL include directories: ${DGL_INCLUDE_DIRS}")
  message(STATUS "DGL link directories: ${DGL_INCLUDE_DIRS}")
  if(MSVC)
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/dgl_sparse/build.bat BUILD_SCRIPT)
    add_custom_target(
      dgl_sparse
      ALL
      ${CMAKE_COMMAND} -E env
      CMAKE_COMMAND=${CMAKE_CMD}
      CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
      USE_CUDA=${USE_CUDA}
      BINDIR=${BINDIR}
      INCLUDEDIR="${DGL_INCLUDE_DIRS}"
      CFLAGS=${CMAKE_C_FLAGS}
      CXXFLAGS=${CMAKE_CXX_FLAGS}
      LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS}
      cmd /e:on /c ${BUILD_SCRIPT} ${TORCH_PYTHON_INTERPS}
      DEPENDS ${BUILD_SCRIPT}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dgl_sparse)
  else(MSVC)
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/dgl_sparse/build.sh BUILD_SCRIPT)
    add_custom_target(
      dgl_sparse
      ALL
      ${CMAKE_COMMAND} -E env
      CMAKE_COMMAND=${CMAKE_CMD}
      CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
      USE_CUDA=${USE_CUDA}
      BINDIR=${CMAKE_CURRENT_BINARY_DIR}
      INCLUDEDIR="${DGL_INCLUDE_DIRS}"
      CFLAGS=${CMAKE_C_FLAGS}
      CXXFLAGS=${CMAKE_CXX_FLAGS}
      LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS}
      bash ${BUILD_SCRIPT} ${TORCH_PYTHON_INTERPS}
      DEPENDS ${BUILD_SCRIPT}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dgl_sparse)
  endif(MSVC)
  add_dependencies(dgl_sparse dgl)
endif(BUILD_SPARSE)

if(BUILD_GRAPHBOLT)
  message(STATUS "Configuring graphbolt library")
  file(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR} BINDIR)
  file(TO_NATIVE_PATH ${CMAKE_COMMAND} CMAKE_CMD)
  if(MSVC)
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/graphbolt/build.bat BUILD_SCRIPT)
    add_custom_target(
      graphbolt
      ALL
      ${CMAKE_COMMAND} -E env
      CMAKE_COMMAND=${CMAKE_CMD}
      BINDIR=${BINDIR}
      CFLAGS=${CMAKE_C_FLAGS}
      CXXFLAGS=${CMAKE_CXX_FLAGS}
      LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS}
      cmd /e:on /c ${BUILD_SCRIPT} ${TORCH_PYTHON_INTERPS}
      DEPENDS ${BUILD_SCRIPT}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/graphbolt)
  else(MSVC)
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/graphbolt/build.sh BUILD_SCRIPT)
    add_custom_target(
      graphbolt
      ALL
      ${CMAKE_COMMAND} -E env
      CMAKE_COMMAND=${CMAKE_CMD}
      BINDIR=${CMAKE_CURRENT_BINARY_DIR}
      CFLAGS=${CMAKE_C_FLAGS}
      CXXFLAGS=${CMAKE_CXX_FLAGS}
      LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS}
      bash ${BUILD_SCRIPT} ${TORCH_PYTHON_INTERPS}
      DEPENDS ${BUILD_SCRIPT}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/graphbolt)
  endif(MSVC)
endif(BUILD_GRAPHBOLT)
