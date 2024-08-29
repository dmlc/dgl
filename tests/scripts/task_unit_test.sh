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
export DGLTESTDEV=$2
export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=tests:${PWD}/python:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}/_download
export TF_FORCE_GPU_ALLOW_GROWTH=true
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE

if [ $2 == "gpu" ] 
then
  export CUDA_VISIBLE_DEVICES=0
else
  export CUDA_VISIBLE_DEVICES=-1
fi

conda activate ${DGLBACKEND}-ci

python3 -m pip install expecttest

if [ $DGLBACKEND == "mxnet" ]
then
  python3 -m pytest -v --junitxml=pytest_compute.xml --durations=100 --ignore=tests/python/common/test_ffi.py tests/python/common || fail "common"
else
  python3 -m pytest -v --junitxml=pytest_dgl_import.xml tests/python/test_dgl_import.py || fail "dgl_import"
  python3 -m pytest -v --junitxml=pytest_common.xml --durations=100 tests/python/common || fail "common"
fi
python3 -m pytest -v --junitxml=pytest_backend.xml --durations=100 tests/python/$DGLBACKEND || fail "backend-specific"
