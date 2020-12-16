#!/bin/bash

set -e

. /opt/conda/etc/profile.d/conda.sh

pip install -r /asv/torch_gpu_pip.txt
pip install pandas rdflib ogb


# install
pushd python
rm -rf build *.egg-info dist
pip uninstall -y dgl
python3 setup.py install
popd
