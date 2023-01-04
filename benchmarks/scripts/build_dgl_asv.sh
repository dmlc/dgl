#!/bin/bash

set -e

# Default building only with cpu
DEVICE=${DGL_BENCH_DEVICE:-cpu}

pip install -r /asv/torch_gpu_pip.txt
pip install pandas rdflib ogb

# build
CMAKE_VARS="-DUSE_OPENMP=ON"
if [[ $DEVICE == "gpu" ]]; then
    CMAKE_VARS="-DUSE_CUDA=ON -DUSE_NCCL=ON $CMAKE_VARS"
fi
arch=`uname -m`
if [[ $arch == *"x86"* ]]; then
  CMAKE_VARS="-DUSE_AVX=ON $CMAKE_VARS"
fi
mkdir -p build
pushd build
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DBUILD_TORCH=ON $CMAKE_VARS ..
make -j
popd
