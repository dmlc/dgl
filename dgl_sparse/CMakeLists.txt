# Find PyTorch cmake files and PyTorch versions with the python interpreter $TORCH_PYTHON_INTERPS
# ("python3" or "python" if empty)
if(NOT TORCH_PYTHON_INTERPS)
  find_program(TORCH_PYTHON_INTERPS NAMES python3 python)
endif()
message(STATUS "Using Python interpreter: ${TORCH_PYTHON_INTERPS}")
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/find_cmake.py FIND_CMAKE_PY)
execute_process(
  COMMAND ${TORCH_PYTHON_INTERPS} ${FIND_CMAKE_PY}
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
target_link_libraries(${LIB_DGL_SPARSE_NAME} dmlc)
set(GOOGLE_TEST 0) # Turn off dmlc-core test


target_include_directories(${LIB_DGL_SPARSE_NAME} PRIVATE ${DGL_INCLUDE})
target_link_libraries(${LIB_DGL_SPARSE_NAME} dgl)
