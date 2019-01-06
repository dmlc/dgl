#!/bin/bash

function fail {
    echo FAIL: $@
    exit -1
}

function usage {
    echo "Usage: $0 backend"
}

if [ $# -ne 1 ]; then
    usage
    fail "Error: must specify backend"
fi

BACKEND=$1
export PYTHONPATH=tests:$PYTHONPATH

python3 -m nose -v --with-xunit tests/$BACKEND || fail "backend-specific"
python3 -m nose -v --with-xunit tests/graph_index || fail "graph_index"
python3 -m nose -v --with-xunit tests/compute || fail "compute"
