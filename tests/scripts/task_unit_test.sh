#!/bin/bash

function fail {
    echo FAIL: $@
    exit 1
}

function usage {
    echo "Usage: $0 backend"
}

if [ $# -ne 1 ]; then
    usage
    fail "Error: must specify backend"
fi

BACKEND=$1
export DGLBACKEND=$1
export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=tests:${PWD}/python:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}

#python3 -m nose -v --with-xunit tests/$BACKEND || fail "backend-specific"
#python3 -m nose -v --with-xunit tests/graph_index || fail "graph_index"
python3 -m nose -v --nocapture --with-xunit tests/compute/test_sampler.py:test_10neighbor_sampler_all || fail "compute"
