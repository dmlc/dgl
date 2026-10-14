# "Copyright Advanced Micro Devices, Inc.
# Licensed under the Apache License Version 2.0"

# HIP Module
if(USE_HIP)
  enable_language(HIP)
  find_package(hip REQUIRED)
  find_package(rocthrust REQUIRED)
  find_package(hipcub REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(hipsparse REQUIRED)
  find_package(hiprand REQUIRED)
else(USE_HIP)
  return()
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17"   SUPPORT_CXX17)

################################################################################################
# Config hip compilation 
# Usage:
#  dgl_config_hip(linker_libs)
macro(dgl_config_hip linker_libs)
  set(USE_OPENMP ON CACHE BOOL "HIP requires OpenMP" FORCE)
  find_package(OpenMP REQUIRED)
  list(APPEND ${linker_libs} OpenMP::OpenMP_CXX)
  message(STATUS "Build with OpenMP.")

  include_directories(${HIP_INCLUDE_DIRS})
  option(DGL_USE_HIP "Enable HIP" ON)
  add_definitions(-DHIP_ENABLE_WARP_SYNC_BUILTINS)
  add_definitions(-DHIPBLAS_USE_HIP_HALF)

  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -I${HIP_INCLUDE_DIRS}")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wunused-result -w")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_AMD__")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${HIP_INCLUDE_DIRS}")
  message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  message(STATUS "CMAKE_HIP_FLAGS: ${CMAKE_HIP_FLAGS}")

  list(APPEND ROCM_HIPCC_FLAGS --offload-arch=${CMAKE_HIP_ARCHITECTURES})
  list(APPEND ROCM_HIPCC_FLAGS "--expt-extended-lambda;-Wno-deprecated-declarations;-std=c++17")
  message(STATUS "ROCM_HIPCC_FLAGS: ${ROCM_HIPCC_FLAGS}")

  #list(APPEND ${linker_libs} 
  #	  ${HIPBLAS_LIBRARIES}
  #	  ${HIPSPARSE_LIBRARIES}
  #	  ${HIPCUB_LIBRARIES}
  #	  ${HIPRAND_LIBRARIES})

  message(STATUS "linker_libs: ${linker_libs}")
endmacro()
