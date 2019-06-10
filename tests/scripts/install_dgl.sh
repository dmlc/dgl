#!/bin/bash
set -e

pushd python
rm -rf build *.egg-info dist
pip3 uninstall -y dgl
# test install
python3 setup.py install
# test inplace build (for cython)
python3 setup.py build_ext --inplace
popd
