cmake_minimum_required(VERSION 3.5)
project(tensoradapter_torch C CXX)

execute_process(
  COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE)
set(Torch_DIR "${TORCH_PREFIX}/Torch")
find_package(Torch REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TORCH_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "CFLAGS ${TORCH_C_FLAGS}")
message(STATUS "CXX_FLAGS ${TORCH_CXX_FLAGS}")
file(GLOB TA_TORCH_SRC *.cpp)
add_library(tensoradapter_torch SHARED "${TA_TORCH_SRC}")
target_include_directories(tensoradapter_torch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_include_directories(tensoradapter_torch PRIVATE "${TORCH_INCLUDE_DIRS}")
target_link_libraries(tensoradapter_torch PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET tensoradapter_torch PROPERTY CXX_STANDARD 14)
