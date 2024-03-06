cmake_minimum_required(VERSION 3.18)
project(graphbolt C CXX)
set (CMAKE_CXX_STANDARD 17)

if(USE_CUDA)
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

set(BOLT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(BOLT_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
file(GLOB BOLT_HEADERS ${BOLT_INCLUDE})
file(GLOB BOLT_SRC ${BOLT_DIR}/*.cc)
if(USE_CUDA)
  file(GLOB BOLT_CUDA_SRC
    ${BOLT_DIR}/cuda/*.cu
    ${BOLT_DIR}/cuda/*.cc
  )
  list(APPEND BOLT_SRC ${BOLT_CUDA_SRC})
  if(DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDAARCHS})
  endif()
endif()

add_library(${LIB_GRAPHBOLT_NAME} SHARED ${BOLT_SRC} ${BOLT_HEADERS})
target_include_directories(${LIB_GRAPHBOLT_NAME} PRIVATE ${BOLT_DIR}
                           ${BOLT_HEADERS}
                           "../third_party/dmlc-core/include"
                           "../third_party/pcg/include")
target_link_libraries(${LIB_GRAPHBOLT_NAME} "${TORCH_LIBRARIES}")
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  target_include_directories(${LIB_GRAPHBOLT_NAME} PRIVATE "../third_party/liburing/src/include")
  get_filename_component(PARENT_DIR "${CMAKE_SOURCE_DIR}" DIRECTORY)
  target_link_libraries(${LIB_GRAPHBOLT_NAME} ${PARENT_DIR}/build/third_party/liburing/lib/liburing.a)
endif()

if(USE_CUDA)
  set_target_properties(${LIB_GRAPHBOLT_NAME} PROPERTIES CUDA_STANDARD 17)
  message(STATUS "Use external CCCL library for a consistent API and performance for graphbolt.")
  target_include_directories(${LIB_GRAPHBOLT_NAME} PRIVATE
                             "../third_party/cccl/thrust"
                             "../third_party/cccl/cub"
                             "../third_party/cccl/libcudacxx/include")

  message(STATUS "Use HugeCTR gpu_cache for graphbolt with INCLUDE_DIRS $ENV{GPU_CACHE_INCLUDE_DIRS}.")
  target_include_directories(${LIB_GRAPHBOLT_NAME} PRIVATE $ENV{GPU_CACHE_INCLUDE_DIRS})
  target_link_directories(${LIB_GRAPHBOLT_NAME} PRIVATE ${GPU_CACHE_BUILD_DIR})
  target_link_libraries(${LIB_GRAPHBOLT_NAME} gpu_cache)
  
  get_property(archs TARGET ${LIB_GRAPHBOLT_NAME} PROPERTY CUDA_ARCHITECTURES)
  message(STATUS "CUDA_ARCHITECTURES for graphbolt: ${archs}")
endif()

# The Torch CMake configuration only sets up the path for the MKL library when
# using the conda distribution. The following is a workaround to address this
# when using a standalone installation of MKL.
if(DEFINED MKL_LIBRARIES)
  target_link_directories(${LIB_GRAPHBOLT_NAME} PRIVATE
                          ${MKL_ROOT}/lib/${MKL_ARCH})
endif()
