#!/bin/bash

set -e

DEVICE=$1
ROOT=/asv/dgl

. /opt/conda/etc/profile.d/conda.sh

# only bench pytorch backend
conda activate "pytorch-ci"

pip install --upgrade pip
pip install asv numpy pandas
pip uninstall -y dgl

# build
BUILD_DIR=/tmp/dgl-build$RANDOM
CMAKE_VARS="-DUSE_CUDA=ON"
export DGL_LIBRARY_PATH=$BUILD_DIR
mkdir -p $BUILD_DIR
pushd $BUILD_DIR
cmake $CMAKE_VARS $ROOT
make -j
popd

# install
pushd $ROOT/python
rm -rf build *.egg-info dist
pip uninstall -y dgl
python3 setup.py install
rm -rf build *.egg-info dist
popd

# benchmark
export DGL_BENCH_DEVICE=$DEVICE
pushd $ROOT/benchmarks
cat asv.conf.json
ls -lh
asv machine --yes
asv run -e --python=same --verbose
asv publish
popd
