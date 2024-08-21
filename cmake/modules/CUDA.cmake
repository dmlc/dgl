# CUDA Module
if(USE_CUDA)
  find_cuda(${USE_CUDA} REQUIRED)
else(USE_CUDA)
  return()
endif()

###### Borrowed from MSHADOW project

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17"   SUPPORT_CXX17)

set(dgl_known_gpu_archs "35" "50" "60" "70" "75")
set(dgl_cuda_arch_ptx "70")
if (CUDA_VERSION_MAJOR GREATER_EQUAL "11")
  list(APPEND dgl_known_gpu_archs "80" "86")
  set(dgl_cuda_arch_ptx "80" "86")
endif()
if (CUDA_VERSION VERSION_GREATER_EQUAL "11.8")
  list(APPEND dgl_known_gpu_archs "89" "90")
  set(dgl_cuda_arch_ptx "90")
endif()
if (CUDA_VERSION VERSION_GREATER_EQUAL "12.0")
  list(REMOVE_ITEM dgl_known_gpu_archs "35")
endif()

################################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   dgl_detect_installed_gpus(out_variable)
function(dgl_detect_installed_gpus out_variable)
set(CUDA_gpu_detect_output "")
  if(NOT CUDA_gpu_detect_output)
    message(STATUS "Running GPU architecture autodetection")
    set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

    file(WRITE ${__cufile} ""
      "#include <cstdio>\n"
      "#include <iostream>\n"
      "using namespace std;\n"
      "int main()\n"
      "{\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) { return -1; }\n"
      "  if (count == 0) { cerr << \"No cuda devices detected\" << endl; return -1; }\n"
      "  for (int device = 0; device < count; ++device)\n"
      "  {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")
    if(MSVC)
      #find vcvarsall.bat and run it building msvc environment
      get_filename_component(MY_COMPILER_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
      find_file(MY_VCVARSALL_BAT vcvarsall.bat "${MY_COMPILER_DIR}/.." "${MY_COMPILER_DIR}/../..")
      execute_process(COMMAND ${MY_VCVARSALL_BAT} && ${CUDA_NVCC_EXECUTABLE} -arch native --run  ${__cufile}
                      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                      RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
    else()
      if(CUDA_LIBRARY_PATH)
        set(CUDA_LINK_LIBRARY_PATH "-L${CUDA_LIBRARY_PATH}")
      endif()
      execute_process(COMMAND ${CUDA_NVCC_EXECUTABLE} -arch native --run ${__cufile} ${CUDA_LINK_LIBRARY_PATH}
                      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                      RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    if(__nvcc_res EQUAL 0)
      # nvcc outputs text containing line breaks when building with MSVC.
      # The line below prevents CMake from inserting a variable with line
      # breaks in the cache
      message(STATUS "Found GPU arch ${__nvcc_out}")
      string(REGEX MATCH "([1-9].[0-9])" __nvcc_out "${__nvcc_out}")
      if(__nvcc_out VERSION_LESS "3.5")
        # drop support for cc < 3.5 and build for all known archs.
        message(WARNING "GPU arch less than 3.5 is not supported.")
      else()
        set(CUDA_gpu_detect_output ${__nvcc_out} CACHE INTERNAL "Returned GPU architetures from mshadow_detect_gpus tool" FORCE)
      endif()
    else()
      message(WARNING "Running GPU detection script with nvcc failed: ${__nvcc_out}")
    endif()
  endif()

  if(NOT CUDA_gpu_detect_output)
    message(WARNING "Automatic GPU detection failed. Building for all known architectures (${dgl_known_gpu_archs}).")
    set(${out_variable} ${dgl_known_gpu_archs} PARENT_SCOPE)
  else()
    set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
  endif()
endfunction()


################################################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_NAME
# Usage:
#   dgl_select_nvcc_arch_flags(out_variable)
function(dgl_select_nvcc_arch_flags out_variable)
  # List of arch names. Turing and Ada don't have a new major version, so they are not added to default build.
  set(__archs_names "Kepler" "Maxwell" "Pascal" "Volta" "Turing" "Ampere" "Ada" "Hopper" "All" "Manual")
  if (NOT CUDA_VERSION VERSION_LESS "12.0")
    list(REMOVE_ITEM __archs_names "Kepler")
  endif()
  set(__archs_name_default "All")
  if(NOT CMAKE_CROSSCOMPILING)
    list(APPEND __archs_names "Auto")
    set(__archs_name_default "Auto")
  endif()

  # set CUDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
  set(CUDA_ARCH_NAME ${__archs_name_default} CACHE STRING "Select target NVIDIA GPU achitecture.")
  set_property( CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${__archs_names} )
  mark_as_advanced(CUDA_ARCH_NAME)

  # verify CUDA_ARCH_NAME value
  if(NOT ";${__archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
    string(REPLACE ";" ", " __archs_names "${__archs_names}")
    message(FATAL_ERROR "Only ${__archs_names} architeture names are supported.")
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(CUDA_ARCH_BIN ${dgl_known_gpu_archs} CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
    set(CUDA_ARCH_PTX ${dgl_cuda_arch_ptx} CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for")
    mark_as_advanced(CUDA_ARCH_BIN CUDA_ARCH_PTX)
  else()
    unset(CUDA_ARCH_BIN CACHE)
    unset(CUDA_ARCH_PTX CACHE)
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Kepler")
    set(__cuda_arch_bin "35")
    set(__cuda_arch_ptx "35")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
    set(__cuda_arch_bin "50")
    set(__cuda_arch_ptx "50")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Pascal")
    set(__cuda_arch_bin "60")
    set(__cuda_arch_ptx "60")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Volta")
    set(__cuda_arch_bin "70")
    set(__cuda_arch_ptx "70")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Turing")
    set(__cuda_arch_bin "75")
    set(__cuda_arch_ptx "75")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Ampere")
    set(__cuda_arch_bin "80")
    set(__cuda_arch_ptx "80")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Ada")
    set(__cuda_arch_bin "89")
    set(__cuda_arch_ptx "89")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Hopper")
    set(__cuda_arch_bin "90")
    set(__cuda_arch_ptx "90")
  elseif(${CUDA_ARCH_NAME} STREQUAL "All")
    set(__cuda_arch_bin ${dgl_known_gpu_archs})
    set(__cuda_arch_ptx ${dgl_cuda_arch_ptx})
  elseif(${CUDA_ARCH_NAME} STREQUAL "Auto")
    dgl_detect_installed_gpus(__cuda_arch_bin)
    # if detect successes, __cuda_arch_ptx = __cuda_arch_bin
    # if detect fails, __cuda_arch_ptx is the latest arch in __cuda_arch_bin
    list(GET __cuda_arch_bin -1 __cuda_arch_ptx)
  else()  # (${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(__cuda_arch_bin ${CUDA_ARCH_BIN})
    set(__cuda_arch_ptx ${CUDA_ARCH_PTX})
  endif()

  # remove dots and convert to lists
  string(REGEX REPLACE "\\." "" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX REPLACE "\\." "" __cuda_arch_ptx "${__cuda_arch_ptx}")
  string(REGEX MATCHALL "[0-9()]+" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+"   __cuda_arch_ptx "${__cuda_arch_ptx}")
  mshadow_list_unique(__cuda_arch_bin __cuda_arch_ptx)

  set(__nvcc_flags "--expt-relaxed-constexpr")
  set(__nvcc_archs_readable "")
  set(__archs "")

  # Tell NVCC to add binaries for the specified GPUs
  foreach(__arch ${__cuda_arch_bin})
    if(__arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified PTX for the concrete BIN
      list(APPEND __nvcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      list(APPEND __nvcc_archs_readable sm_${CMAKE_MATCH_1})
      list(APPEND __archs ${CMAKE_MATCH_1})
    else()
      # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
      list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=sm_${__arch})
      list(APPEND __nvcc_archs_readable sm_${__arch})
      list(APPEND __archs ${__arch})
    endif()
  endforeach()

  # Tell NVCC to add PTX intermediate code for the specified architectures
  foreach(__arch ${__cuda_arch_ptx})
    list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=compute_${__arch})
    list(APPEND __nvcc_archs_readable compute_${__arch})
  endforeach()

  string(REPLACE ";" " " __nvcc_archs_readable "${__nvcc_archs_readable}")
  set(${out_variable}          ${__nvcc_flags}          PARENT_SCOPE)
  set(${out_variable}_readable ${__nvcc_archs_readable} PARENT_SCOPE)
  set(CUDA_ARCHITECTURES       ${__archs}               PARENT_SCOPE)
endfunction()

################################################################################################
# Config cuda compilation and append CUDA libraries to linker_libs
# Usage:
#  dgl_config_cuda(linker_libs)
macro(dgl_config_cuda linker_libs)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Cannot find CUDA.")
  endif()
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
	include_directories(${CUDA_INCLUDE_DIRS})

  add_definitions(-DDGL_USE_CUDA)

  # NVCC flags
  # Manually set everything
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)

  # 0. Add host flags
  message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "[ \t\n\r]" "," CXX_HOST_FLAGS "${CMAKE_CXX_FLAGS}")
  if(MSVC AND NOT USE_MSVC_MT)
    string(CONCAT CXX_HOST_FLAGS ${CXX_HOST_FLAGS} ",/MD")
  endif()
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "${CXX_HOST_FLAGS}")
  if(USE_OPENMP)
    # Needed by CUDA disjoint union source file.
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "${OpenMP_CXX_FLAGS}")
  endif(USE_OPENMP)

  # 1. Add arch flags
  dgl_select_nvcc_arch_flags(NVCC_FLAGS_ARCH)
  list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_ARCH})

  # 2. flags in third_party/moderngpu
  list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda;-Wno-deprecated-declarations;-std=c++17")

  message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

  list(APPEND ${linker_libs} 
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cusparse_LIBRARY})
endmacro()
