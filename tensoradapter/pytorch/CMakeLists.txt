cmake_minimum_required(VERSION 3.5)
project(tensoradapter_pytorch C CXX)

execute_process(
  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/find_cmake.py
  OUTPUT_VARIABLE TORCH_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE)
set(Torch_DIR "${TORCH_PREFIX}/Torch")
find_package(Torch REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TORCH_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -ggdb")
message(STATUS "CFLAGS ${TORCH_C_FLAGS}")
message(STATUS "CXX_FLAGS ${TORCH_CXX_FLAGS}")
file(GLOB TA_TORCH_SRC *.cpp)
add_library(tensoradapter_pytorch SHARED "${TA_TORCH_SRC}")
target_include_directories(tensoradapter_pytorch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_include_directories(tensoradapter_pytorch PRIVATE "${TORCH_INCLUDE_DIRS}")
target_link_libraries(tensoradapter_pytorch PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET tensoradapter_pytorch PROPERTY CXX_STANDARD 14)
