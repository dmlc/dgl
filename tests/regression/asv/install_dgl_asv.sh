#!/bin/bash

set -e

python -m pip install numpy pandas pytest

. /opt/conda/etc/profile.d/conda.sh

# only use pytorch
conda activate "pytorch-ci"
pushd python
rm -rf build *.egg-info dist
pip uninstall -y dgl
python3 setup.py install
popd
