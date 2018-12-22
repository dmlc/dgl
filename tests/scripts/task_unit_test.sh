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

python3 -m nose -v --with-xunit tests || fail "tests"
python3 -m nose -v --with-xunit tests/$BACKEND || fail "backend"
python3 -m nose -v --with-xunit tests/graph_index || fail "graph_index"
