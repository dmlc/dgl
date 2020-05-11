#!/bin/bash

# Argument
#  - dev: cpu or gpu
if [ $# -ne 1 ]; then
    echo "Device argument required, can be cpu or gpu"
    exit -1
fi

dev=$1

set -e
. /opt/conda/etc/profile.d/conda.sh

rm -rf _deps
mkdir _deps
pushd _deps
conda activate "pytorch-ci"
if [ "$dev" == "gpu" ]; then
  pip uninstall -y dgl
  pip install --pre dgl
  python3 setup.py install
else
  pip uninstall -y dgl-cu101
  pip install --pre dgl-cu101
  python3 setup.py install
fi
popd