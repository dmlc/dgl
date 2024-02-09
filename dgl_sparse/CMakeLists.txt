cmake_minimum_required(VERSION 3.8)
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

set(SPARSE_LINKER_LIBS "")

if(USE_CUDA)
  add_definitions(-DDGL_USE_CUDA)
  enable_language(CUDA)
endif()

# For windows, define NOMINMAX to avoid conflict with std::min/max
if(MSVC)
  add_definitions(-DNOMINMAX)
endif()

set(Torch_DIR "${TORCH_PREFIX}/Torch")
message(STATUS "Setting directory to ${Torch_DIR}")
find_package(Torch REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TORCH_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -ggdb")

set(LIB_DGL_SPARSE_NAME "dgl_sparse_pytorch_${TORCH_VER}")
list(APPEND SPARSE_LINKER_LIBS ${TORCH_LIBRARIES})

set(SPARSE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(SPARSE_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
file(GLOB SPARSE_HEADERS ${SPARSE_INCLUDE})
file(GLOB SPARSE_SRC
  ${SPARSE_DIR}/*.cc
  ${SPARSE_DIR}/cpu/*.cc
)
if(USE_CUDA)
  file(GLOB SPARSE_CUDA_SRC
    ${SPARSE_DIR}/cuda/*.cu
  )
  list(APPEND SPARSE_SRC ${SPARSE_CUDA_SRC})
endif()

add_library(${LIB_DGL_SPARSE_NAME} SHARED ${SPARSE_SRC} ${SPARSE_HEADERS})
target_include_directories(
  ${LIB_DGL_SPARSE_NAME} PRIVATE ${SPARSE_DIR} ${SPARSE_HEADERS})
target_link_libraries(${LIB_DGL_SPARSE_NAME} ${SPARSE_LINKER_LIBS})
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
if (EXTERNAL_DMLC_LIB_PATH)
   # external dmlc requires OpenMP link
   include(FindOpenMP)
   if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
   endif(OPENMP_FOUND)	
   message(STATUS "looking for dmlc library in ${EXTERNAL_DMLC_LIB_PATH}")
   find_package(dmlc REQUIRED HINTS ${EXTERNAL_DMLC_LIB_PATH})
   target_link_libraries(${LIB_DGL_SPARSE_NAME} dmlc::dmlc dgl)
else (EXTERNAL_DMLC_LIB_PATH)
   target_link_libraries(${LIB_DGL_SPARSE_NAME} dmlc dgl)
endif()
set(GOOGLE_TEST 0) # Turn off dmlc-core test

# Configure dgl_sparse library to use C++17 standard for compatibility with PyTorch
set_property(TARGET ${LIB_DGL_SPARSE_NAME} PROPERTY CXX_STANDARD 17)
