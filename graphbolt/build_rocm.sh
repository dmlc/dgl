#"Copyright Advanced Micro Devices, Inc.
#Licensed under the Apache License Version 2.0"

#!/bin/bash
# Helper script to build graphbolt libraries for PyTorch
set -euo pipefail
GRAPHBOLT_SRCDIR=$(dirname $0)
# We build directly in our primary build directory
GRAPHBOLT_BINDIR=$BINDIR/graphbolt
GRAPHBOLT_BUILD_DIR=$GRAPHBOLT_BINDIR

mkdir -p $GRAPHBOLT_BUILD_DIR
cd $GRAPHBOLT_BUILD_DIR

if [ $(uname) = 'Darwin' ]; then
  CPSOURCE=*.dylib
else
  CPSOURCE=*.so
fi

# Use compilers from parent CMake if provided
CMAKE_COMPILER_FLAGS=""
if [ -n "$CMAKE_C_COMPILER" ]; then
  CMAKE_COMPILER_FLAGS="$CMAKE_COMPILER_FLAGS -DCMAKE_C_COMPILER=$CMAKE_C_COMPILER"
fi
if [ -n "$CMAKE_CXX_COMPILER" ]; then
  CMAKE_COMPILER_FLAGS="$CMAKE_COMPILER_FLAGS -DCMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER"
fi

# Use CMAKE_PREFIX_PATH from parent if provided (helps find ROCm components)
CMAKE_PREFIX_FLAGS=""
if [ -n "$CMAKE_PREFIX_PATH" ]; then
  CMAKE_PREFIX_FLAGS="-DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"
fi

CMAKE_FLAGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON                                \
-DCMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}                           \
-DUSE_HIP=$USE_HIP                                                             \
-DGPU_TARGETS=${CMAKE_HIP_ARCHITECTURES} $CMAKE_COMPILER_FLAGS $CMAKE_PREFIX_FLAGS "
echo "graphbolt cmake flags: $CMAKE_FLAGS"

if [ $# -eq 0 ]; then
  $CMAKE_COMMAND $CMAKE_FLAGS -S $GRAPHBOLT_SRCDIR -B $GRAPHBOLT_BUILD_DIR
  cmake --build $GRAPHBOLT_BUILD_DIR --parallel
else
  for PYTHON_INTERP in $@; do
    TORCH_VER=$($PYTHON_INTERP -c 'import torch; print(torch.__version__.split("+")[0])')
    mkdir -p $TORCH_VER
    cd $TORCH_VER
    $CMAKE_COMMAND $CMAKE_FLAGS -DPYTHON_INTERP=$PYTHON_INTERP -S $GRAPHBOLT_SRCDIR -B $GRAPHBOLT_BUILD_DIR
    cmake --build $GRAPHBOLT_BUILD_DIR --parallel
    cd ..
  done
fi
