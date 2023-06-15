cmake_minimum_required(VERSION 3.5)
project(tensoradapter_pytorch C CXX)

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

if(USE_CUDA)
  add_definitions(-DDGL_USE_CUDA)
endif()

set(Torch_DIR "${TORCH_PREFIX}/Torch")
message(STATUS "Setting directory to ${Torch_DIR}")
find_package(Torch REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TORCH_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -ggdb")
set(TORCH_TARGET_NAME "tensoradapter_pytorch_${TORCH_VER}")
file(GLOB TA_TORCH_SRC *.cpp)
add_library(${TORCH_TARGET_NAME} SHARED "${TA_TORCH_SRC}")

# use the library name rather than the path
set(TENSORADAPTER_TORCH_LIBS torch)

message(STATUS "tensoradapter found PyTorch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "tensoradapter found PyTorch lib: ${TENSORADAPTER_TORCH_LIBS}")

target_include_directories(
  ${TORCH_TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_include_directories(
  ${TORCH_TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/dlpack/include")
target_include_directories(
  ${TORCH_TARGET_NAME} PRIVATE "${TORCH_INCLUDE_DIRS}")
target_link_libraries(${TORCH_TARGET_NAME} PRIVATE "${TENSORADAPTER_TORCH_LIBS}")
set_property(TARGET ${TORCH_TARGET_NAME} PROPERTY CXX_STANDARD 17)
message(STATUS "Configured target ${TORCH_TARGET_NAME}")
