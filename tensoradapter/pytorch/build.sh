#!/bin/bash
# Helper script to build tensor adapter libraries for PyTorch
set -e

mkdir -p build
mkdir -p $BINDIR/tensoradapter/pytorch
cd build

if [ $(uname) = 'Darwin' ]; then
	CPSOURCE=*.dylib
else
	CPSOURCE=*.so
fi

CMAKE_FLAGS="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST -DUSE_CUDA=$USE_CUDA"

if [ $# -eq 0 ]; then
	$CMAKE_COMMAND $CMAKE_FLAGS ..
	make -j
	cp -v $CPSOURCE $BINDIR/tensoradapter/pytorch
else
	for PYTHON_INTERP in $@; do
		TORCH_VER=$($PYTHON_INTERP -c 'import torch; print(torch.__version__.split("+")[0])')
		mkdir -p $TORCH_VER
		cd $TORCH_VER
		$CMAKE_COMMAND $CMAKE_FLAGS -DPYTHON_INTERP=$PYTHON_INTERP ../..
		make -j
		cp -v $CPSOURCE $BINDIR/tensoradapter/pytorch
		cd ..
	done
fi
