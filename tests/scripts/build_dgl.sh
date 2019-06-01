#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Device argument required, can be cpu or gpu"
    exit -1
fi

if [ "$1" == "cuda" ]; then
    cp cmake/config.cmake config.cmake
    sed -i -e 's/USE_CUDA OFF/USE_CUDA ON/g' config.cmake
fi

if [ -d build ]; then
	rm -rf build
fi
mkdir build

rm -rf _download

pushd build
cmake ..
make -j4
popd

pushd python
rm -rf build *.egg-info dist
pip3 uninstall -y dgl
# test install
python3 setup.py install
# test inplace build (for cython)
python3 setup.py build_ext --inplace
popd
