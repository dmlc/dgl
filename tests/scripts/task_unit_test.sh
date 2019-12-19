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
export DGL_DOWNLOAD_DIR=${PWD}
export TF_FORCE_GPU_ALLOW_GROWTH=true

conda activate ${DGLBACKEND}-ci

python3 -m pytest -v --junitxml=pytest_compute.xml tests/compute || fail "compute"
python3 -m pytest -v --junitxml=pytest_gindex.xml tests/graph_index || fail "graph_index"
python3 -m pytest -v --junitxml=pytest_backend.xml tests/$DGLBACKEND || fail "backend-specific"

export OMP_NUM_THREADS=1
if [ $2 != "gpu" ] && [ $1 != "tensorflow"]; then
    python3 -m pytest -v --junitxml=pytest_distributed.xml tests/distributed || fail "distributed"
fi
