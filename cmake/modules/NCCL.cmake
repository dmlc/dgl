include(ExternalProject)

# set path to submodule
set(NCCL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/nccl")

# NCCL doesn't have CMAKE, so build externally
ExternalProject_Add(nccl_external
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/nccl
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND
    env
    make
    "src.build"
    "-j"
    "BUILDDIR=${NCCL_BUILD_DIR}"
  BUILD_BYPRODUCTS "${NCCL_BUILD_DIR}/lib/libnccl_static.a"
  INSTALL_COMMAND ""
  )

# set output variables
set(NCCL_FOUND TRUE)
set(NCCL_LIBRARY "${NCCL_BUILD_DIR}/lib/libnccl_static.a")
set(NCCL_INCLUDE_DIR "${NCCL_BUILD_DIR}/include")
