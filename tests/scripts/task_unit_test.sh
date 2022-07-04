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

TEST_DEVICE=$2
if [[ $2 == "cugraph" ]]; then
    TEST_DEVICE=gpu
fi
export DGLBACKEND=$1
export DGLTESTDEV=${TEST_DEVICE}
export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=tests:${PWD}/python:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}
export TF_FORCE_GPU_ALLOW_GROWTH=true

if [[ $2 == "gpu" || $2 == "cugraph" ]]
then
  export CUDA_VISIBLE_DEVICES=0
else
  export CUDA_VISIBLE_DEVICES=-1
fi

if [[ $2 != "cugraph" ]]; then
    conda activate ${DGLBACKEND}-ci
fi

python3 -m pip install pytest psutil pyyaml pydantic pandas rdflib ogb || fail "pip install"

if [[ $2 == "cugraph" ]]; then
    python3 -m pytest -v --junitxml=pytest_cugraph.xml --durations=20 tests/cugraph || fail "cugraph"
else
    python3 -m pytest -v --junitxml=pytest_compute.xml --durations=100 tests/compute || fail "compute"
    python3 -m pytest -v --junitxml=pytest_backend.xml --durations=100 tests/$DGLBACKEND || fail "backend-specific"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export DMLC_LOG_DEBUG=1
if [ $2 == "cpu" && $DGLBACKEND == "pytorch" ]; then
    python3 -m pip install filelock
    python3 -m pytest -v --capture=tee-sys --junitxml=pytest_distributed.xml --durations=100 tests/distributed/*.py || fail "distributed"
    PYTHONPATH=tools:$PYTHONPATH python3 -m pytest -v --capture=tee-sys --junitxml=pytest_tools.xml tests/tools/*.py || fail "tools"
fi
