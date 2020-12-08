#!/bin/bash

set -e

. /opt/conda/etc/profile.d/conda.sh

# only bench pytorch backend
conda activate "pytorch-ci"
python -m pip install numpy pandas pytest

pushd python
rm -rf build *.egg-info dist
pip uninstall -y dgl
python3 setup.py install
popd
