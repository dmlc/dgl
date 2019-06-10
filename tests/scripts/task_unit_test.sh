#!/bin/bash

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

python3 -m nose -v --with-xunit tests/compute || fail "compute"
python3 -m nose -v --with-xunit tests/graph_index || fail "graph_index"
python3 -m nose -v --with-xunit tests/$DGLBACKEND || fail "backend-specific"
export OMP_NUM_THREADS=1
if [ $2 != "gpu" ]; then
    python3 -m nose -v --with-xunit tests/distributed || fail "distributed"
fi
