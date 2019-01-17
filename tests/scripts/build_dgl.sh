#!/bin/bash

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
python3 setup.py install
popd
