#!/bin/bash
set -e
. /opt/conda/etc/profile.d/conda.sh

if [ $# -ne 1 ]; then
    echo "Device argument required, can be cpu, gpu or cugraph"
    exit -1
fi

if [[ $1 != "cpu" ]]; then
    # CI is now running on g4dn instance. Specify target arch to avoid below
    # error: Unknown CUDA Architecture Name 9.0a in CUDA_SELECT_NVCC_ARCH_FLAGS
    export TORCH_CUDA_ARCH_LIST=7.5 # For dgl_sparse and tensoradaptor.
    CMAKE_VARS="$CMAKE_VARS -DUSE_CUDA=ON -DCUDA_ARCH_NAME=Turing" # For graphbolt.
fi

# This is a semicolon-separated list of Python interpreters containing PyTorch.
# The value here is for CI.  Replace it with your own or comment this whole
# statement for default Python interpreter.
if [ "$1" != "cugraph" ]; then
    # We do not build pytorch for cugraph because currently building
    # pytorch against all the supported cugraph versions is not supported
    # See issue: https://github.com/rapidsai/cudf/issues/8510
    CMAKE_VARS="$CMAKE_VARS -DTORCH_PYTHON_INTERPS=/opt/conda/envs/pytorch-ci/bin/python"
else
    # Disable sparse build as cugraph docker image lacks cuDNN.
    CMAKE_VARS="$CMAKE_VARS -DBUILD_TORCH=OFF -DBUILD_SPARSE=OFF"
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
    DGLBACKEND=${backend} python3 setup.py install
    # test inplace build (for cython)
    DGLBACKEND=${backend} python3 setup.py build_ext --inplace
    done
fi
popd
