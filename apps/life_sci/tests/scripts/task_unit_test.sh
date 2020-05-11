#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh

function fail {
    echo FAIL: $@
    exit -1
}

function usage {
    echo "Usage: $0 backend device"
}

if [ $# -ne 2 ]; then
    usage
    fail "Error: must specify backend and device"
fi

export DGLBACKEND=$1
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}

if [ $2 == "gpu" ]
then
  export CUDA_VISIBLE_DEVICES=0
else
  export CUDA_VISIBLE_DEVICES=-1
fi

conda activate ${DGLBACKEND}-ci
pip install _deps/dgl*.whl

python3 -m pytest -v --junitxml=pytest_data.xml tests/data || fail "data"
python3 -m pytest -v --junitxml=pytest_model.xml tests/model || fail "model"
python3 -m pytest -v --junitxml=pytest_utils.xml tests/utils || fail "utils"