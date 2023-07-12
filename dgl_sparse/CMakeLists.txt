cmake_minimum_required(VERSION 3.5)
project(dgl_sparse C CXX)

# Find PyTorch cmake files and PyTorch versions with the python interpreter $PYTHON_INTERP
# ("python3" or "python" if empty)
if(NOT PYTHON_INTERP)
  find_program(PYTHON_INTERP NAMES python3 python)
endif()
message(STATUS "Using Python interpreter: ${PYTHON_INTERP}")
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/find_cmake.py FIND_CMAKE_PY)
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_CMAKE_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_cmake.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)
message(STATUS "Configuring for PyTorch ${TORCH_VER}")
string(REPLACE "." ";" TORCH_VERSION_LIST ${TORCH_VER})
list(GET TORCH_VERSION_LIST 0 TORCH_VERSION_MAJOR)
list(GET TORCH_VERSION_LIST 1 TORCH_VERSION_MINOR)

if(USE_CUDA)
  add_definitions(-DDGL_USE_CUDA)
endif()

set(Torch_DIR "${TORCH_PREFIX}/Torch")
message(STATUS "Setting directory to ${Torch_DIR}")
find_package(Torch REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TORCH_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -ggdb")

set(LIB_DGL_SPARSE_NAME "dgl_sparse_pytorch_${TORCH_VER}")

set(SPARSE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(SPARSE_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
file(GLOB SPARSE_HEADERS ${SPARSE_INCLUDE})
file(GLOB SPARSE_SRC
  ${SPARSE_DIR}/*.cc
)
add_library(${LIB_DGL_SPARSE_NAME} SHARED ${SPARSE_SRC} ${SPARSE_HEADERS})
target_include_directories(
  ${LIB_DGL_SPARSE_NAME} PRIVATE ${SPARSE_DIR} ${SPARSE_HEADERS})
target_link_libraries(${LIB_DGL_SPARSE_NAME} "${TORCH_LIBRARIES}")
target_compile_definitions(${LIB_DGL_SPARSE_NAME} PRIVATE TORCH_VERSION_MAJOR=${TORCH_VERSION_MAJOR})
target_compile_definitions(${LIB_DGL_SPARSE_NAME} PRIVATE TORCH_VERSION_MINOR=${TORCH_VERSION_MINOR})

target_include_directories(${LIB_DGL_SPARSE_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/third_party/dmlc-core/include")
message(STATUS "DGL include directories: ${DGL_INCLUDE_DIRS}")
target_include_directories(${LIB_DGL_SPARSE_NAME} PRIVATE ${DGL_INCLUDE_DIRS})
target_link_directories(${LIB_DGL_SPARSE_NAME} PRIVATE ${DGL_BUILD_DIR} "${DGL_BUILD_DIR}/third_party/dmlc-core")

# The Torch CMake configuration only sets up the path for the MKL library when
# using the conda distribution. The following is a workaround to address this
# when using a standalone installation of MKL.
if(DEFINED MKL_LIBRARIES)
  target_link_directories(${LIB_DGL_SPARSE_NAME} PRIVATE ${MKL_ROOT}/lib/${MKL_ARCH})
endif()

target_link_libraries(${LIB_DGL_SPARSE_NAME} dmlc dgl)
set(GOOGLE_TEST 0) # Turn off dmlc-core test

# Configure dgl_sparse library to use C++17 standard for compatibility with PyTorch
set_property(TARGET ${LIB_DGL_SPARSE_NAME} PROPERTY CXX_STANDARD 17)
