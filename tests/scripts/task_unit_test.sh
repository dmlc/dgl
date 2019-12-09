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

conda activate ${DGLBACKEND}-ci
export LD_LIBRARY_PATH_OLD=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
pip install pytest
python3 -m pytest tests/compute || fail "compute"
python3 -m pytest tests/graph_index || fail "graph_index"
python3 -m pytest tests/$DGLBACKEND || fail "backend-specific"
export OMP_NUM_THREADS=1
if [ $2 != "gpu" ]; then
    python3 -m pytest tests/distributed || fail "distributed"
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_OLD