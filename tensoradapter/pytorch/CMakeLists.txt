cmake_minimum_required(VERSION 3.5)
project(tensoradapter_pytorch C CXX)

function(add_pytorch_target CONDA_ENV)
  # Find PyTorch cmake files and PyTorch versions within CONDA_ENV (or the current environment
  # if CONDA_ENV is empty).
  if(NOT MSVC)
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/find_cmake_ver.sh FIND_CMAKE_VER_SH)
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/find_cmake.py FIND_CMAKE_PY)
    execute_process(
      COMMAND bash ${FIND_CMAKE_VER_SH} ${FIND_CMAKE_PY} ${CONDA_ENV}
      OUTPUT_VARIABLE TORCH_PREFIX_VER
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  else()
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/find_cmake_ver.bat FIND_CMAKE_VER_SH)
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/find_cmake.py FIND_CMAKE_PY)
    execute_process(
      COMMAND "${FIND_CMAKE_VER_SH}" ${FIND_CMAKE_PY} ${CONDA_ENV}
      OUTPUT_VARIABLE TORCH_PREFIX_VER
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
  list(GET TORCH_PREFIX_VER 1 TORCH_VER)
  if(CONDA_ENV)
    message(STATUS "Configuring for PyTorch ${TORCH_VER} for conda environment ${CONDA_ENV}")
  else(CONDA_ENV)
    message(STATUS "Configuring for PyTorch ${TORCH_VER}")
  endif(CONDA_ENV)

  set(Torch_DIR "${TORCH_PREFIX}/Torch")
  find_package(Torch REQUIRED)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TORCH_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -ggdb")
  set(TORCH_TARGET_NAME "tensoradapter_pytorch_${TORCH_VER}")
  file(GLOB TA_TORCH_SRC *.cpp)
  add_library(${TORCH_TARGET_NAME} SHARED "${TA_TORCH_SRC}")
  target_include_directories(${TORCH_TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
  target_include_directories(${TORCH_TARGET_NAME} PRIVATE "${TORCH_INCLUDE_DIRS}")
  target_link_libraries(${TORCH_TARGET_NAME} PRIVATE "${TORCH_LIBRARIES}")
  set_property(TARGET ${TORCH_TARGET_NAME} PROPERTY CXX_STANDARD 14)
  message(STATUS "Configured target ${TORCH_TARGET_NAME}")

  add_custom_command(
    TARGET ${TORCH_TARGET_NAME} POST_BUILD COMMAND
    cmd.exe /c "COPY /Y Release\\${TORCH_TARGET_NAME}.dll .")
endfunction()

if(TORCH_CONDA_ENVS)
  foreach(CONDA_ENV ${TORCH_CONDA_ENVS})
    add_pytorch_target(${CONDA_ENV})
  endforeach(CONDA_ENV)
else()
  add_pytorch_target("")
endif()
