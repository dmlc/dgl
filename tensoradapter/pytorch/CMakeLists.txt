cmake_minimum_required(VERSION 3.5)
project(tensoradapter_pytorch C CXX)

function(add_pytorch_target PYTHON_INTERP)
  # Find PyTorch cmake files and PyTorch versions with the python interpreter $PYTHON_INTERP
  # (or "python" if empty)
  file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/find_cmake.py FIND_CMAKE_PY)
  if(NOT PYTHON_INTERP)
    set(PYTHON_INTERP python)
  endif()
  message(STATUS "Using Python interpreter: ${PYTHON_INTERP}")
  execute_process(
    COMMAND ${PYTHON_INTERP} ${FIND_CMAKE_PY}
    OUTPUT_VARIABLE TORCH_PREFIX_VER
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "find_cmake.py output: ${TORCH_PREFIX_VER}")
  list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
  list(GET TORCH_PREFIX_VER 1 TORCH_VER)
  message(STATUS "Configuring for PyTorch ${TORCH_VER}")

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

  if(MSVC)
    add_custom_command(
      TARGET ${TORCH_TARGET_NAME} POST_BUILD COMMAND
      cmd.exe /c "COPY /Y Release\\${TORCH_TARGET_NAME}.dll .")
  endif(MSVC)
endfunction()

if(TORCH_PYTHON_INTERPS)
  foreach(PYTHON_INTERP ${TORCH_PYTHON_INTERPS})
    add_pytorch_target(${PYTHON_INTERP})
  endforeach(PYTHON_INTERP)
else()
  add_pytorch_target("")
endif()
