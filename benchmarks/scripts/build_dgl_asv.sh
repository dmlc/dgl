#!/bin/bash

set -e

. /opt/conda/etc/profile.d/conda.sh
conda activate pytorch-ci
# Default building only with cpu
DEVICE=${DGL_BENCH_DEVICE:-cpu}

# build
if [[ $DEVICE == "cpu" ]]; then
    CMAKE_VARS=""
else
    CMAKE_VARS="-DUSE_CUDA=ON"
fi
mkdir -p build
pushd build
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DBUILD_TORCH=ON $CMAKE_VARS ..
make -j
popd

conda deactivate
