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

[ $1 == "pytorch" ] || fail "Distrbuted tests run on pytorch backend only."
[ $2 == "cpu" ] || fail "Distrbuted tests run on cpu only."

export DGLBACKEND=$1
export DGLTESTDEV=$2
export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=tests:${PWD}/python:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}/_download
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE

export CUDA_VISIBLE_DEVICES=-1

conda activate ${DGLBACKEND}-ci

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export DMLC_LOG_DEBUG=1

python3 -m pytest -v --capture=tee-sys --junitxml=pytest_distributed.xml --durations=100 tests/distributed/*.py || fail "distributed"

PYTHONPATH=tools:tools/distpartitioning:$PYTHONPATH python3 -m pytest -v --capture=tee-sys --junitxml=pytest_tools.xml --durations=100 tests/tools/*.py || fail "tools"
