cmake_minimum_required(VERSION 3.5)
project(tensoradapter_torch C CXX)

execute_process(
  COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE)
set(Torch_DIR "${TORCH_PREFIX}/Torch")
message(STATUS "Find torch directory in ${Torch_DIR}")
find_package(Torch REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Torch_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Torch_CXX_FLAGS}")
file(GLOB TA_TORCH_SRC *.cpp)
add_library(tensoradapter_torch SHARED "${TA_TORCH_SRC}")
target_include_directories(tensoradapter_torch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(tensoradapter_torch PRIVATE "${Torch_LIBRARIES}")
set_property(TARGET tensoradapter_torch PROPERTY CXX_STANDARD 14)
