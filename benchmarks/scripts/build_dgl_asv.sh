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

/asv/dgl/benchmarks/env/06715dc6ea55f2b02b6ebc21bca7e34a/lib/python3.10/site-packages
for t_dir in /asv/dgl/benchmarks/env/*; do
  cp -R /asv/dgl/benchmarks/benchmarks "${t_dir}/lib/python3.10/site-packages/"
done
