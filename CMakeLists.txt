cmake_minimum_required(VERSION 3.18)
########################################
# Borrowed and adapted from TVM project
########################################
project(dgl C CXX)
message(STATUS "Start configuring project ${PROJECT_NAME}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cmake utils
include(cmake/util/Util.cmake)
include(cmake/util/MshadowUtil.cmake)
include(cmake/util/FindCUDA.cmake)

# Options for building DGL.
# NOTE: Please avoid editing this file to change build type. Instead, using
# bash script/build_dgl.sh -e -t release to overwrite the value.
dgl_option(BUILD_TYPE "Type of the build: dev, dogfood or release" "dev")
message(STATUS "Build for ${BUILD_TYPE}")

dgl_option(USE_CUDA "Build with CUDA" OFF)
dgl_option(TORCH_PYTHON_INTERPS "Python interpreter for building sub-components" python3)

# Conda build related options.
dgl_option(EXTERNAL_DLPACK_PATH "Path to external dlpack" OFF)
dgl_option(EXTERNAL_DMLC_PATH "Path to external dmlc-core" OFF)
dgl_option(EXTERNAL_DMLC_LIB_PATH "Path to external dmlc-core library" OFF)
dgl_option(EXTERNAL_PHMAP_PATH "Path to external parallel-hashmap" OFF)
dgl_option(EXTERNAL_NANOFLANN_PATH "Path to use external nanoflann" OFF)
dgl_option(EXTERNAL_METIS_PATH "Path to external metis" OFF)
dgl_option(EXTERNAL_METIS_LIB_PATH "Path to external metis library" OFF)
dgl_option(EXTERNAL_GKLIB_PATH "Path to external gklib" OFF)

# Options for building DGL features: "none," "dev," "dogfood," "release," and
# "all."
#    "none"  - The feature is OFF for all build types. This is used when
#              disabling a feature.
#    "dev"   - The feature is ON for dev build. The default build from source
#              and the build for unit tests are using this build type.
#  "dogfood" - The major function of this feature is done. The regression and
#              benchmark framework are using this build type.
#  "release" - The feature will be build for release.
#    "all"   - The feature is ON for all build types. Equivalent to set ["dev"
#              "dogfood" "release"].
# NOTE: Please avoid editing this file to change feature options for a local
# build. Instead, using bash script/build_dgl.sh -e '-DFEATURE_NAME=ON/OFF' to
# overwrite the value.
dgl_feature_option(
    BUILD_SPARSE
    "Build DGL sparse library"
    "all"
)
dgl_feature_option(
    BUILD_TORCH
    "Build the PyTorch plugin"
    "all"
)
dgl_feature_option(
    USE_EPOLL
    "Build with epoll for socket communicator"
    "all"
)
dgl_feature_option(
    USE_LIBXSMM
    "Build with LIBXSMM library optimization"
    "all"
)
dgl_feature_option(
    USE_OPENMP
    "Build with OpenMP"
    "all"
)

dgl_feature_option(
    BUILD_GRAPHBOLT
    "Build Graphbolt library"
    "all"
)

dgl_feature_option(
    LIBCXX_ENABLE_PARALLEL_ALGORITHMS
    "Enable the parallel algorithms library. This requires the PSTL to be available."
    "none"
)
dgl_feature_option(
    REBUILD_LIBXSMM
    "Clean LIBXSMM build cache at every build"
    "none"
)
dgl_feature_option(
    USE_HDFS
    "Build with HDFS support"
    "none"
) # Set env HADOOP_HDFS_HOME if needed
dgl_feature_option(
    USE_S3
    "Build with S3 support"
    "none"
)

# Only build C++ tests for unit testing purposes in dev build.
dgl_feature_option(
    BUILD_CPP_TEST
    "Build cpp unittest executables"
    "dev"
)

if (EXTERNAL_DLPACK_PATH OR EXTERNAL_DMLC_PATH OR EXTERNAL_NANOFLANN_PATH OR EXTERNAL_NANOFLANN_PATH OR EXTERNAL_METIS_PATH OR EXTERNAL_GKLIB_PATH)
  message(STATUS "Using at least one external library")
  set(USE_EXTERNAL_LIBS ON)
  
  if (BUILD_CPP_TEST)
    message(FATAL_ERROR "Cannot build cpp unittests with external libraries")
  endif(BUILD_CPP_TEST)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# Set optimization options for different build types.
if (${BUILD_TYPE} STREQUAL "dev")
  if (MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Od")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od")
  else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g3 -ggdb")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g3 -ggdb")
  endif()
else()
  if (MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /DNDEBUG")
  else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG")
  endif()
endif()

if(USE_CUDA)
  message(STATUS "Build with CUDA support")
  project(dgl C CXX)
  include(cmake/modules/CUDA.cmake)
  message(STATUS "Use external CCCL library for a consistent API and performance.")
  cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust")
  cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/cccl/cub")
  cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include")
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
  set(CMAKE_C_FLAGS "-Wall -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-Wall -fPIC ${CMAKE_CXX_FLAGS}")
  if(NOT APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--warn-common ${CMAKE_SHARED_LINKER_FLAGS}")
  endif(NOT APPLE)
endif(MSVC)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  message(STATUS "Disabling LIBXSMM on ${CMAKE_SYSTEM_PROCESSOR}.")
  set(USE_LIBXSMM OFF)
endif()

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

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  list(APPEND DGL_LINKER_LIBS OpenMP::OpenMP_CXX)
  message(STATUS "Build with OpenMP.")
endif(USE_OPENMP)

# Configure cuda
if(USE_CUDA)
  file(GLOB_RECURSE DGL_CUDA_SRC
    src/array/cuda/*.cc
    src/array/cuda/*.cu
    src/array/cuda/uvm/*.cc
    src/array/cuda/uvm/*.cu
    src/kernel/cuda/*.cc
    src/kernel/cuda/*.cu
    src/partition/cuda/*.cu
    src/runtime/cuda/*.cc
    src/runtime/cuda/*.cu
    src/geometry/cuda/*.cu
    src/graph/transform/cuda/*.cu
    src/graph/sampling/randomwalks/*.cu
  )
  list(APPEND DGL_SRC ${DGL_CUDA_SRC})
  dgl_config_cuda(DGL_LINKER_LIBS)
  cuda_add_library(dgl SHARED ${DGL_SRC})
else(USE_CUDA)
  add_library(dgl SHARED ${DGL_SRC})
endif(USE_CUDA)

if ((NOT MSVC) AND USE_EPOLL)
  INCLUDE(CheckIncludeFile)
  check_include_file("sys/epoll.h" EPOLL_AVAILABLE)
  if (EPOLL_AVAILABLE)
    target_compile_definitions(dgl PRIVATE USE_EPOLL)
  else()
    message(WARNING "EPOLL is not available on this platform...")
  endif()
endif ()

# include directories
target_include_directories(dgl PRIVATE "include")
# check for conda includes
if("$ENV{CONDA_BUILD}" STREQUAL "1")
  set(in_conda_build TRUE)
  message(STATUS "Conda build environment detected")
elseif(DEFINED ENV{CONDA_PREFIX})
  set(in_conda_prefix TRUE)
  message(STATUS "Conda environment detected: $ENV{CONDA_PREFIX}")
endif()

if (USE_CONDA_INCLUDES)
  if(in_conda_build)
    message(STATUS "Using Conda build environment includes: $ENV{PREFIX}")
    target_include_directories(dgl PRIVATE "$ENV{PREFIX}/include" "$ENV{BUILD_PREFIX}/include")
  elseif(in_conda_prefix)
    message(STATUS "Using Conda environment includes: $ENV{CONDA_PREFIX}")
    target_include_directories(dgl PRIVATE "$ENV{CONDA_PREFIX}/include")
  else()
    message(FATAL_ERROR "Conda environment not detected")
  endif()
endif()

if(EXTERNAL_DLPACK_PATH)
  message(STATUS "looking for dlpack headers in ${EXTERNAL_DLPACK_PATH}")
  include_directories(SYSTEM ${EXTERNAL_DLPACK_PATH})
else(EXTERNAL_DLPACK_PATH)
  target_include_directories(dgl PRIVATE "third_party/dlpack/include")
endif(EXTERNAL_DLPACK_PATH)

if(EXTERNAL_DMLC_PATH)
  if (USE_HDFS)
    message(FATAL_ERROR "Cannot use HDFS and external dmlc-core at the same time")
  endif()
  message(STATUS "looking for dmlc headers in ${EXTERNAL_DMLC_PATH}")
  include_directories(SYSTEM ${EXTERNAL_DMLC_PATH})
  
  if (NOT EXTERNAL_DMLC_LIB_PATH)
    message(FATAL_ERROR "EXTERNAL_DMLC_LIB_PATH must be set if EXTERNAL_DMLC_PATH is set")
  endif()
  message(STATUS "looking for dmlc library in ${EXTERNAL_DMLC_LIB_PATH}")
  find_package(dmlc
    REQUIRED
    HINTS ${EXTERNAL_DMLC_LIB_PATH}
  )
  if(NOT dmlc_FOUND)
      message(FATAL_ERROR "Failed to find DMLC library")
  endif()
  list(APPEND DGL_LINKER_LIBS dmlc::dmlc)

else(EXTERNAL_DMLC_PATH)
  target_include_directories(dgl PRIVATE "third_party/dmlc-core/include")
  # For serialization
  if (USE_HDFS)
    option(DMLC_HDFS_SHARED "dgl has to build with dynamic hdfs library" ON)
  endif()
  add_subdirectory("third_party/dmlc-core")
  list(APPEND DGL_LINKER_LIBS dmlc)
  set(GOOGLE_TEST 0) # Turn off dmlc-core test
endif(EXTERNAL_DMLC_PATH)

target_include_directories(dgl PRIVATE "tensoradapter/include")
target_include_directories(dgl PRIVATE "third_party/pcg/include")
target_include_directories(dgl PRIVATE "third_party/tsl_robin_map/include")

if(EXTERNAL_NANOFLANN_PATH)
  include_directories(SYSTEM ${EXTERNAL_NANOFLANN_PATH})
else(EXTERNAL_NANOFLANN_PATH)
  target_include_directories(dgl PRIVATE "third_party/nanoflann/include")
endif(EXTERNAL_NANOFLANN_PATH)

if (USE_LIBXSMM)
  target_compile_definitions(dgl PRIVATE USE_LIBXSMM DGL_CPU_LLC_SIZE=40000000 __BLAS=0)
  target_include_directories(dgl PRIVATE "third_party/libxsmm/include")
  message(STATUS "Build with LIBXSMM optimization.")
endif()

# To compile METIS correct for DGL.
add_compile_definitions(IDXTYPEWIDTH=64 REALTYPEWIDTH=32)
if (EXTERNAL_METIS_PATH)
  # To compile METIS correct for DGL.
  if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DIDXTYPEWIDTH=64 /DREALTYPEWIDTH=32")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DIDXTYPEWIDTH=64 /DREALTYPEWIDTH=32")
  else(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=32")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=32")
  endif(MSVC)
  find_package(METIS REQUIRED)
  message(STATUS "Found METIS library")
  target_include_directories(dgl SYSTEM PUBLIC ${METIS_INCLUDE_DIR})
  list(APPEND DGL_LINKER_LIBS ${METIS_LIBRARIES})
else(EXTERNAL_METIS_PATH)
  target_include_directories(dgl PRIVATE "third_party/METIS/include")
  # Compile METIS
  if(NOT MSVC)
    set(GKLIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/GKlib")
    include(${GKLIB_PATH}/GKlibSystem.cmake)
    include_directories(${GKLIB_PATH})
    add_library(GKlib ${GKlib_sources})
    include_directories("third_party/METIS/include/")
    add_subdirectory("third_party/METIS/libmetis/")
    # When building on ubi7, it fails with the following error:
    # /usr/include/signal.h:156:29: error: unknown type name 'siginfo_t'.
    # So I(Rui) define _POSIX_C_SOURCE to 200809L for GKlib and metis to avoid the error.
    target_compile_definitions(GKlib PRIVATE _POSIX_C_SOURCE=200809L)
    target_compile_definitions(metis PRIVATE _POSIX_C_SOURCE=200809L)
    list(APPEND DGL_LINKER_LIBS metis GKlib)
  endif(NOT MSVC)
endif(EXTERNAL_METIS_PATH)


# Avoid exposing third-party symbols when using DGL as a library.
if((NOT MSVC) AND (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--exclude-libs,ALL")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--exclude-libs,ALL")
endif()

# Compile gpu_cache
if(USE_CUDA)
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
  target_compile_definitions(dgl PRIVATE PARALLEL_ALGORITHMS)
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
      EXTERNAL_DMLC_LIB_PATH=${EXTERNAL_DMLC_LIB_PATH}
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
      EXTERNAL_DMLC_LIB_PATH=${EXTERNAL_DMLC_LIB_PATH}
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
  include_directories("third_party/tsl_robin_map/include")
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
  string(REPLACE ";" "\\;" CUDA_ARCHITECTURES_ESCAPED "${CUDA_ARCHITECTURES}")
  file(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR} BINDIR)
  file(TO_NATIVE_PATH ${CMAKE_COMMAND} CMAKE_CMD)
  if(MSVC)
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/graphbolt/build.bat BUILD_SCRIPT)
    add_custom_target(
      graphbolt
      ALL
      ${CMAKE_COMMAND} -E env
      CMAKE_COMMAND=${CMAKE_CMD}
      CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
      USE_CUDA=${USE_CUDA}
      BINDIR=${BINDIR}
      CFLAGS=${CMAKE_C_FLAGS}
      CXXFLAGS=${CMAKE_CXX_FLAGS}
      CUDAARCHS="${CUDA_ARCHITECTURES_ESCAPED}"
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
      CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
      USE_CUDA=${USE_CUDA}
      USE_LIBURING=${USE_LIBURING}
      BINDIR=${CMAKE_CURRENT_BINARY_DIR}
      CFLAGS=${CMAKE_C_FLAGS}
      CXXFLAGS=${CMAKE_CXX_FLAGS}
      CUDAARCHS="${CUDA_ARCHITECTURES_ESCAPED}"
      LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS}
      bash ${BUILD_SCRIPT} ${TORCH_PYTHON_INTERPS}
      DEPENDS ${BUILD_SCRIPT}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/graphbolt)
  endif(MSVC)
endif(BUILD_GRAPHBOLT)
