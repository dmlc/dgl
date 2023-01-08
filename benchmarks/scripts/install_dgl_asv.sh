#!/bin/bash

set -e

# install
pushd python
rm -rf build *.egg-info dist
pip uninstall -y dgl
python3 setup.py install
popd
