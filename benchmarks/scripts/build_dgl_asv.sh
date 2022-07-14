#!/bin/bash

set -e

# . /opt/conda/etc/profile.d/conda.sh
# conda activate pytorch-ci
# Default building only with cpu
DEVICE=${DGL_BENCH_DEVICE:-cpu}

pip install -r /asv/torch_gpu_pip.txt
pip install pandas rdflib ogb

# build
if [[ $DEVICE == "cpu" ]]; then
    CMAKE_VARS=""
else
    CMAKE_VARS="-DUSE_CUDA=ON"
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

# conda deactivate
