#!/bin/bash

GCN_EXAMPLE_DIR="../../examples/pytorch/"

function fail {
    echo FAIL: $@
    exit -1
}

function usage {
    echo "Usage: $0 [cpu|cuda]"
}

# check arguments
if [ $# -ne 1 ]; then
    usage
    fail "Error: must specify device"
fi

if [ "$1" == "cpu" ]; then
    dev=-1
elif [ "$1" == "cuda" ]; then
    export CUDA_VISIBLE_DEVICES=0
    dev=0
else
    usage
    fail "Unknown device $1"
fi

pushd $GCN_EXAMPLE_DIR> /dev/null

# test
python3 pagerank.py || fail "run pagerank.py on $1"
python3 gcn/gcn.py --dataset cora --gpu $dev || fail "run gcn/gcn.py on $1"
python3 gcn/gcn_spmv.py --dataset cora --gpu $dev || fail "run gcn/gcn_spmv.py on $1"

popd > /dev/null
