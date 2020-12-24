#!/bin/bash
# Helper script to build tensor adapter libraries for PyTorch
set -e

rm -rf build
mkdir -p build
mkdir -p $BINDIR/tensoradapter/pytorch
cd build

if [ $# -eq 0 ]; then
	cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR ..
	make -j
	cp -v *.so $BINDIR/tensoradapter/pytorch
else
	for PYTHON_INTERP in $@; do
		rm -rf *
		cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR -DPYTHON_INTERP=$PYTHON_INTERP ..
		make -j
		cp -v *.so $BINDIR/tensoradapter/pytorch
	done
fi
