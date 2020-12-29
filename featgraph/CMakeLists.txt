cmake_minimum_required(VERSION 3.5)

project(featgraph C CXX)
message(STATUS "Start configuring project ${PROJECT_NAME}")

# Find CUDA
include(../cmake/util/FindCUDA.cmake)
find_cuda(ON)
message(STATUS "${CUDA_INCLUDE_DIRS}")

add_custom_target(
  featgraph_kernel
  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/pack_featgraph.py
)

set(CMAKE_CXX_FLAGS "-std=c++14 -O2 -fPIC")
file(GLOB FEATGRAPH_SRC
  src/featgraph.cc
  src/tvm_runtime_pack.cc
)
add_library(featgraph SHARED ${FEATGRAPH_SRC})
target_include_directories(featgraph PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(featgraph PRIVATE "../third_party/tvm/include")
target_include_directories(featgraph PRIVATE "../third_party/tvm/3rdparty/dmlc-core/include")
target_include_directories(featgraph PRIVATE "../third_party/tvm/3rdparty/dlpack/include")
target_include_directories(featgraph PRIVATE "./include")
target_link_libraries(featgraph "dl"  # dynamic linking
                                ${CUDA_CUDART_LIBRARY}
                                ${CUDA_CUDA_LIBRARY}
                                ${CUDA_NVRTC_LIBRARY})
add_dependencies(featgraph featgraph_kernel)

install(TARGETS featgraph LIBRARY DESTINATION lib)
