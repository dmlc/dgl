cmake_minimum_required(VERSION 3.5)

project(tensoradapter C CXX)
message(STATUS "Start configuring project ${PROJECT_NAME}")

if(BUILD_TORCH)
  add_subdirectory("pytorch")
endif(BUILD_TORCH)
