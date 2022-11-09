#!/bin/bash
# Helper script to build dgl sparse libraries
set -e

rm -rf build
mkdir -p build
mkdir -p $BINDIR/dgl_sparse
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
	cp -v $CPSOURCE $BINDIR/dgl_sparse
else
	for PYTHON_INTERP in $@; do
		rm -rf *
		$CMAKE_COMMAND $CMAKE_FLAGS -DPYTHON_INTERP=$PYTHON_INTERP ..
		make -j
		cp -v $CPSOURCE $BINDIR/dgl_sparse
	done
fi
