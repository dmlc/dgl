#!/bin/bash
# Helper script to build dgl sparse libraries for PyTorch
set -e

mkdir -p build
mkdir -p $BINDIR/dgl_sparse
cd build

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

if [ "$USE_CUDA" = 'ON' ]; then
  CMAKE_FLAGS="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST -DUSE_CUDA=$USE_CUDA -DEXTERNAL_DMLC_LIB_PATH=$EXTERNAL_DMLC_LIB_PATH"
elif [ "$USE_HIP" = 'ON' ]; then
  CMAKE_FLAGS="-DPYTORCH_ROCM_ARCH=${CMAKE_HIP_ARCHITECTURES}                  \
  -DGPU_TARGETS=${CMAKE_HIP_ARCHITECTURES}                                     \
  -DCMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}                         \
  -DUSE_HIP=$USE_HIP                                                           \
  -DEXTERNAL_DMLC_LIB_PATH=$EXTERNAL_DMLC_LIB_PATH $CMAKE_COMPILER_FLAGS $CMAKE_PREFIX_FLAGS"
fi
# CMake passes in the list of directories separated by spaces.  Here we replace them with semicolons.
CMAKE_FLAGS="$CMAKE_FLAGS -DDGL_INCLUDE_DIRS=${INCLUDEDIR// /;} -DDGL_BUILD_DIR=$BINDIR"
echo $CMAKE_FLAGS

if [ $# -eq 0 ]; then
	$CMAKE_COMMAND $CMAKE_FLAGS ..
	cmake --build .
	cp -v $CPSOURCE $BINDIR/dgl_sparse
else
	for PYTHON_INTERP in $@; do
		TORCH_VER=$($PYTHON_INTERP -c 'import torch; print(torch.__version__.split("+")[0])')
		mkdir -p $TORCH_VER
		cd $TORCH_VER
		$CMAKE_COMMAND $CMAKE_FLAGS -DPYTHON_INTERP=$PYTHON_INTERP ../..
		cmake --build .
		cp -v $CPSOURCE $BINDIR/dgl_sparse
		cd ..
	done
fi
