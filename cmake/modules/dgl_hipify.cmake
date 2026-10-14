#"Copyright Advanced Micro Devices, Inc.
# Licensed under the Apache License Version 2.0"

# cmake file to trigger hipify

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

function(hipify)
  set(flags)
  set(singleValueArgs CUDA_SOURCE_DIR HIP_SOURCE_DIR CONFIG_FILE CUSTOM_MAP_FILE)
  set(multiValueArgs HEADER_INCLUDE_DIR IGNORES)

  cmake_parse_arguments(HIPIFY "${flags}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  find_package(Python3)

  set(HIPIFY_COMMAND
    python3 ${HIPIFY_CUDA_SOURCE_DIR}/dgl_hipify.py
    --project-directory ${HIPIFY_CUDA_SOURCE_DIR}
    --output-directory ${HIPIFY_CUDA_SOURCE_DIR}
  )
  message("Hipify: ${HIPIFY_COMMAND}")

  execute_process(
   COMMAND ${HIPIFY_COMMAND}
   RESULT_VARIABLE hipify_return_value
  )
  if (NOT hipify_return_value EQUAL 0)
    message(FATAL_ERROR "Failed to hipify files!")
  endif()
endfunction()

