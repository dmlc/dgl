#!/bin/bash
set -e
. /opt/conda/etc/profile.d/conda.sh

if [ $# -ne 1 ]; then
    echo "Device argument required, can be cpu, gpu or cugraph"
    exit -1
fi

CMAKE_VARS="-DBUILD_CPP_TEST=ON -DUSE_OPENMP=ON"
# This is a semicolon-separated list of Python interpreters containing PyTorch.
# The value here is for CI.  Replace it with your own or comment this whole
# statement for default Python interpreter.
if [ "$1" != "cugraph" ]; then
    # We do not build pytorch for cugraph because currently building
    # pytorch against all the supported cugraph versions is not supported
    # See issue: https://github.com/rapidsai/cudf/issues/8510
    CMAKE_VARS="$CMAKE_VARS -DBUILD_TORCH=ON -DBUILD_SPARSE=ON -DTORCH_PYTHON_INTERPS=/opt/conda/envs/pytorch-ci/bin/python"
fi

#This is implemented to detect underlying architecture and enable arch specific optimization.
arch=`uname -m`
if [[ $arch == *"x86"* ]]; then
  CMAKE_VARS="-DUSE_AVX=ON $CMAKE_VARS"
fi

if [[ $1 != "cpu" ]]; then
    CMAKE_VARS="-DUSE_CUDA=ON -DUSE_NCCL=ON $CMAKE_VARS"
fi

if [ -d build ]; then
    rm -rf build
fi
mkdir build

rm -rf _download

pushd build
cmake $CMAKE_VARS ..
make -j
popd

pushd python
if [[ $1 == "cugraph" ]]; then
    rm -rf build *.egg-info dist
    pip uninstall -y dgl
    # test install
    python3 setup.py install
    # test inplace build (for cython)
    python3 setup.py build_ext --inplace
else
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
fi
popd
