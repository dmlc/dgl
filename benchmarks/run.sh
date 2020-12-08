#!/bin/bash

set -e

ROOT=/asv/dgl

. /opt/conda/etc/profile.d/conda.sh

# only bench pytorch backend
conda activate "pytorch-ci"

pip install --upgrade pip
pip install asv numpy pandas pytest
pip uninstall -y dgl

# build
pushd $ROOT
rm -rf build
mkdir -p build
CMAKE_VARS="-DUSE_CUDA=ON"
rm -rf _download
pushd build
cmake $CMAKE_VARS ..
make -j
popd
popd

# install
pushd $ROOT/python
rm -rf build *.egg-info dist
pip uninstall -y dgl
python3 setup.py install
popd

# benchmark
pushd $ROOT/benchmarks
cat asv.conf.json
ls -lh
asv machine --yes
asv run --python=existing --verbose
asv publish
popd
