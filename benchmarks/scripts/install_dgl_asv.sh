#!/bin/bash

set -e

# . /opt/conda/etc/profile.d/conda.sh

# install
pushd python
rm -rf build *.egg-info dist
pip uninstall -y dgl
python3 setup.py install
popd
