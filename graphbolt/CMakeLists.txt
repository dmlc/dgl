cmake_minimum_required(VERSION 3.18)
project(graphbolt C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Language support for GPU Hardware Acceleration
if (USE_HIP)
  message(STATUS "Build graphbolt with HIP support")
  enable_language(HIP) 
  # Add macro definitions 
  add_definitions(-DGRAPHBOLT_USE_HIP)
  add_definitions(-DDGL_USE_HIP)
  add_definitions(-DHIP_ENABLE_WARP_SYNC_BUILTINS)

  # Dependencies
  ## ROCm-DS dependencies
  find_package(libhipcxx REQUIRED)
  find_package(cuco REQUIRED)

  ## rocm-libraries dependencies
  find_package(rocprim REQUIRED)
  find_package(hipcub REQUIRED)
  find_package(rocthrust REQUIRED)
  # TODO this is a sledgehammer approach
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
elseif(USE_CUDA)
  message(STATUS "Build graphbolt with CUDA support")
  enable_language(CUDA)
  add_definitions(-DGRAPHBOLT_USE_CUDA)
endif()

# For windows, define NOMINMAX to avoid conflict with std::min/max
if(MSVC)
  add_definitions(-DNOMINMAX)
endif()

# Find PyTorch cmake files and PyTorch versions with the python interpreter
# $PYTHON_INTERP ("python3" or "python" if empty)
if(NOT PYTHON_INTERP)
  find_program(PYTHON_INTERP NAMES python3 python)
endif()

message(STATUS "Using Python interpreter: ${PYTHON_INTERP}")

file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/find_cmake.py FIND_CMAKE_PY)
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_CMAKE_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "find_cmake.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)

message(STATUS "Configuring for PyTorch ${TORCH_VER}")
string(REPLACE "." ";" TORCH_VERSION_LIST ${TORCH_VER})

set(Torch_DIR "${TORCH_PREFIX}/Torch")
message(STATUS "Setting directory to ${Torch_DIR}")

find_package(Torch REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TORCH_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -ggdb")

set(LIB_GRAPHBOLT_NAME "graphbolt_pytorch_${TORCH_VER}")

#
# Additional options
#
option(BUILD_WITH_TASKFLOW "Use taskflow as parallel backend" ON)
option(USE_OPENMP "Use OpenMP for graphbolt" ON)
option(USE_LIBURING "Build graphbolt with liburing support" ON)

#
# Graph Bolt Core Library
#


set(BOLT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(BOLT_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
file(GLOB BOLT_HEADERS ${BOLT_INCLUDE})
file(GLOB BOLT_SRC ${BOLT_DIR}/*.cc)

#
# Configure GPU support
#
if(USE_CUDA)
  file(GLOB BOLT_CUDA_SRC
    ${BOLT_DIR}/cuda/*.cu
    ${BOLT_DIR}/cuda/*.cc
  )
  list(APPEND BOLT_SRC ${BOLT_CUDA_SRC})
  if(DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDAARCHS})
  endif()
  set(CMAKE_CUDA_ARCHITECTURES_FILTERED ${CMAKE_CUDA_ARCHITECTURES})
  # CUDA extension supports only sm_70 and up (Volta+).
  list(FILTER CMAKE_CUDA_ARCHITECTURES_FILTERED EXCLUDE REGEX "[2-6][0-9]")
  list(LENGTH CMAKE_CUDA_ARCHITECTURES_FILTERED CMAKE_CUDA_ARCHITECTURES_FILTERED_LEN)
  if(CMAKE_CUDA_ARCHITECTURES_FILTERED_LEN EQUAL 0)
    # Build the CUDA extension at least build for Volta.
    set(CMAKE_CUDA_ARCHITECTURES_FILTERED "70")
  endif()
  set(LIB_GRAPHBOLT_CUDA_NAME "${LIB_GRAPHBOLT_NAME}_cuda")
elseif(USE_HIP)
  file(GLOB BOLT_CUDA_SRC
    ${BOLT_DIR}/cuda/*.cu
    ${BOLT_DIR}/cuda/*.cc  
  )
  list(APPEND BOLT_SRC ${BOLT_CUDA_SRC})
  set(LIB_GRAPHBOLT_CUDA_NAME "${LIB_GRAPHBOLT_NAME}_hip")
  set_source_files_properties(${BOLT_SRC} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${BOLT_CUDA_SRC} PROPERTIES LANGUAGE HIP)
endif()

# The core graphbolt library.
add_library(${LIB_GRAPHBOLT_NAME} SHARED ${BOLT_SRC} ${BOLT_HEADERS})

# Setup includes

include_directories(BEFORE ${BOLT_DIR}
                           ${BOLT_HEADERS}
                           # For CXX20 features:
                           # `std::atomic_ref`, `std::counting_semaphore`
                           "../third_party/pcg/include"
                           "../third_party/tsl_robin_map/include")
if (USE_CUDA)
# For HIP builds, we include libhipcxx below
include_directories(BEFORE # For CXX20 features:
                           # `std::atomic_ref`, `std::counting_semaphore`
                           "../third_party/cccl/libcudacxx/include")
endif(USE_CUDA)

target_link_libraries(${LIB_GRAPHBOLT_NAME} "${TORCH_LIBRARIES}")
if(BUILD_WITH_TASKFLOW)
  target_include_directories(${LIB_GRAPHBOLT_NAME} PRIVATE "../third_party/taskflow")
  target_compile_definitions(${LIB_GRAPHBOLT_NAME} PRIVATE BUILD_WITH_TASKFLOW=1)
endif()

if(USE_OPENMP OR USE_HIP)
  find_package(OpenMP REQUIRED)
  message(STATUS "OpenMP Found")
  target_link_libraries(${LIB_GRAPHBOLT_NAME} OpenMP::OpenMP_CXX)
  message(STATUS "Build graphbolt with OpenMP.")
endif(USE_OPENMP OR USE_HIP)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  if(USE_LIBURING)
    add_definitions(-DHAVE_LIBRARY_LIBURING)
    message(STATUS "LIBURING Found")
    include(ExternalProject)
    set(LIBURING_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party/liburing)
    set(LIBURING_C_COMPILER "${CMAKE_C_COMPILER} -w")
    ExternalProject_Add(
      liburing
      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/liburing
      CONFIGURE_COMMAND <SOURCE_DIR>/configure --cc=${LIBURING_C_COMPILER} --cxx=${CMAKE_CXX_COMPILER} --prefix=/ --use-libc
      # In order to avoid the error `error: redefinition of 'struct in6_pktinfo'` on ubi7
      # when building examples, let's build src only.
      BUILD_COMMAND bash -c "make -j 4 -C src/"
      BUILD_IN_SOURCE ON
      INSTALL_COMMAND make install DESTDIR=${LIBURING_INSTALL_DIR}
      BUILD_BYPRODUCTS ${LIBURING_INSTALL_DIR}/lib/liburing.a
      BUILD_BYPRODUCTS ${LIBURING_INSTALL_DIR}/include
      DOWNLOAD_EXTRACT_TIMESTAMP true
    )
    set(LIBURING_INCLUDE ${LIBURING_INSTALL_DIR}/include)
    set(LIBURING ${LIBURING_INSTALL_DIR}/lib/liburing.a)

    target_include_directories(${LIB_GRAPHBOLT_NAME} PRIVATE ${LIBURING_INCLUDE})
    add_dependencies(${LIB_GRAPHBOLT_NAME} liburing)
    target_link_libraries(${LIB_GRAPHBOLT_NAME} ${CMAKE_CURRENT_BINARY_DIR}/third_party/liburing/lib/liburing.a)
    message(STATUS "Build graphbolt with liburing.")
  endif(USE_LIBURING)
endif()

#
# GraphBolt Extensions
#
if(USE_CUDA OR USE_HIP)
  message(STATUS "BUild using CUDA_ARCHITECTURES for graphbolt: ${archs}")
  file(GLOB BOLT_CUDA_EXTENSION_SRC
    ${BOLT_DIR}/cuda/extension/*.cu
    ${BOLT_DIR}/cuda/extension/*.cc
    ../third_party/HugeCTR/gpu_cache/src/nv_gpu_cache.cu
  )
  # Until https://github.com/NVIDIA/cccl/issues/1083 is resolved, we need to
  # compile the cuda/extension folder with Volta+ CUDA architectures.
  add_library(${LIB_GRAPHBOLT_CUDA_NAME} STATIC ${BOLT_CUDA_EXTENSION_SRC} ${BOLT_HEADERS})
  target_link_libraries(${LIB_GRAPHBOLT_CUDA_NAME} "${TORCH_LIBRARIES}")

  # General target properties
  set_target_properties(${LIB_GRAPHBOLT_NAME} PROPERTIES CUDA_STANDARD 17)
  set_target_properties(${LIB_GRAPHBOLT_CUDA_NAME} PROPERTIES CUDA_STANDARD 17)
  set_target_properties(${LIB_GRAPHBOLT_CUDA_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

  # Hardware specific target properties
  if (USE_CUDA)
  set_target_properties(${LIB_GRAPHBOLT_CUDA_NAME} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_FILTERED}")
  elseif(USE_HIP)
  set_source_files_properties(${BOLT_CUDA_EXTENSION_SRC} PROPERTIES LANGUAGE HIP)
  set_target_properties(${LIB_GRAPHBOLT_CUDA_NAME} PROPERTIES HIP_STANDARD 17)
  set_target_properties(${LIB_GRAPHBOLT_CUDA_NAME} PROPERTIES LINKER_LANGUAGE HIP)
  endif()

  # Enables libcudacxx/libhipcxx for gpu_cache.
  target_compile_definitions(${LIB_GRAPHBOLT_CUDA_NAME} PRIVATE LIBCUDACXX_VERSION)
  include_directories(AFTER "../third_party/HugeCTR/gpu_cache/include")
  message(STATUS "Build graphbolt extension with HugeCTR GPU embedding cache.")

  # Hardware specific includes and linking
  if (USE_CUDA)
    message(STATUS "Use external CCCL library for a consistent API and performance for graphbolt.")
    include_directories(BEFORE
                        "../third_party/cccl/thrust"
                        "../third_party/cccl/cub"
                        "../third_party/cuco/include")
  elseif(USE_HIP)
    include_directories(
      $<TARGET_PROPERTY:cuco::cuco,INTERFACE_INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:hip::hipcub,INTERFACE_INCLUDE_DIRECTORIES>
      )
    target_link_libraries(${LIB_GRAPHBOLT_CUDA_NAME} 
      cuco::cuco
      hip::device
      hip::host
      hip::hipcub
      roc::rocprim_hip
      roc::rocthrust
    )
  endif()

  # Logging
  if (USE_CUDA)
  get_property(archs TARGET ${LIB_GRAPHBOLT_NAME} PROPERTY CUDA_ARCHITECTURES)
  message(STATUS "CUDA_ARCHITECTURES for graphbolt: ${archs}")

  get_property(archs TARGET ${LIB_GRAPHBOLT_CUDA_NAME} PROPERTY CUDA_ARCHITECTURES)
  message(STATUS "CUDA_ARCHITECTURES for graphbolt extension: ${archs}")

  elseif(USE_HIP)
  get_property(archs TARGET ${LIB_GRAPHBOLT_NAME} PROPERTY HIP_ARCHITECTURES)
  message(STATUS "HIP_ARCHITECTURES for graphbolt: ${archs}")

  get_property(archs TARGET ${LIB_GRAPHBOLT_CUDA_NAME} PROPERTY HIP_ARCHITECTURES)
  message(STATUS "HIP_ARCHITECTURES for graphbolt extension: ${archs}")
  endif()

  # Link to the main library
  target_link_libraries(${LIB_GRAPHBOLT_NAME} ${LIB_GRAPHBOLT_CUDA_NAME})
endif()


# The Torch CMake configuration only sets up the path for the MKL library when
# using the conda distribution. The following is a workaround to address this
# when using a standalone installation of MKL.
if(DEFINED MKL_LIBRARIES)
  target_link_directories(${LIB_GRAPHBOLT_NAME} PRIVATE
                          ${MKL_ROOT}/lib/${MKL_ARCH})
endif()
