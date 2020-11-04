cmake_minimum_required(VERSION 3.5)

project(tensoradapter C CXX)
message(STATUS "Start configuring project ${PROJECT_NAME}")

if(BUILD_TORCH)
  add_subdirectory("torch")
endif(BUILD_TORCH)

file(GLOB TA_SRC *.cpp)
add_library(tensoradapter SHARED "${TA_SRC}")
target_include_directories(tensoradapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(tensoradapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dlpack/include)
target_link_libraries(tensoradapter PRIVATE dl)
set_property(TARGET tensoradapter PROPERTY CXX_STANDARD 14)
