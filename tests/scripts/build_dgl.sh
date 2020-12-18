#!/bin/bash
set -e
. /opt/conda/etc/profile.d/conda.sh

if [ $# -ne 1 ]; then
    echo "Device argument required, can be cpu or gpu"
    exit -1
fi

# This is a semicolon-separated list of conda environments containing PyTorch.
# The value here is for CI.  Replace it with your own environment or uncomment
# this whole statement for default.
TORCH_CONDA_ENVS=pytorch-ci

CMAKE_VARS="-DBUILD_CPP_TEST=ON -DUSE_OPENMP=ON -DBUILD_TORCH=ON -DTORCH_CONDA_ENVS=$TORCH_CONDA_ENVS"

if [ "$1" == "gpu" ]; then
    CMAKE_VARS="-DUSE_CUDA=ON $CMAKE_VARS"
fi

if [ -d build ]; then
    rm -rf build
fi
mkdir build

rm -rf _download

pushd build
cmake $CMAKE_VARS ..
make -j8
popd

pushd python
for backend in pytorch mxnet tensorflow
do 
conda activate "${backend}-ci"
rm -rf build *.egg-info dist
pip uninstall -y dgl
# test install
python3 setup.py install
# test inplace build (for cython)
python3 setup.py build_ext --inplace
done
popd
