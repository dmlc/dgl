#!/bin/bash

set -e

# Default building only with cpu
DEVICE=${DGL_BENCH_DEVICE:-cpu}

pip install -r /asv/torch_gpu_pip.txt

# build
# 'CUDA_TOOLKIT_ROOT_DIR' is always required for sparse build as torch1.13.1+cu116 is installed.
CMAKE_VARS="-DUSE_OPENMP=ON -DBUILD_TORCH=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda"
if [[ $DEVICE == "gpu" ]]; then
    CMAKE_VARS="-DUSE_CUDA=ON $CMAKE_VARS"
fi
mkdir -p build
pushd build
cmake $CMAKE_VARS ..
make -j8
popd
