#!/bin/bash

set -e

. /opt/conda/etc/profile.d/conda.sh

# build
CMAKE_VARS="-DUSE_CUDA=ON"
mkdir -p build
pushd build
cmake $CMAKE_VARS ..
make -j
popd
