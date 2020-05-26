#!/bin/bash

set -e

python -m pip install numpy

. /opt/conda/etc/profile.d/conda.sh

pushd python
for backend in pytorch mxnet tensorflow
do 
conda activate "${backend}-ci"
rm -rf build *.egg-info dist
pip uninstall -y dgl
# test install
python3 setup.py install
# test inplace build (for cython)
python3 setup.py build_ext --inplace
python3 -m pip install -r /root/requirement.txt
done
popd
conda deactivate