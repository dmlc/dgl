#!/bin/bash

set -e

python -m pip install numpy pandas pytest

. /opt/conda/etc/profile.d/conda.sh

for backend in pytorch mxnet tensorflow
do 
  conda activate "${backend}-ci"
  pushd python
  rm -rf build *.egg-info dist
  pip uninstall -y dgl
  python3 setup.py install
  popd
done
conda deactivate
