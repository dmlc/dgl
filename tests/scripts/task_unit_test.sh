#!/bin/bash

function fail {
    echo FAIL: $@
    exit -1
}

function usage {
    echo "Usage: $0 device"
}

if [ $# -ne 1 ]; then
    usage
    fail "Error: must specify device"
fi

DGLTESTDEV=$1
export PYTHONPATH=tests:$PYTHONPATH

python3 -m nose -v --with-xunit tests/$DGLBACKEND || fail "backend-specific"
python3 -m nose -v --with-xunit tests/graph_index || fail "graph_index"
python3 -m nose -v --with-xunit tests/compute || fail "compute"
