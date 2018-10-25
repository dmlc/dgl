#!/bin/bash

GCN_EXAMPLE_DIR="../../examples/pytorch/gcn"

function fail {
    echo FAIL: $@
    exit -1
}

function usage {
    echo "Usage: $0 [CPU|GPU]"
}

# check arguments
if [ $# -ne 1 ]; then
    usage
    fail "Error: must specify device"
fi

if [ "$1" == "CPU" ]; then
    dev=-1
elif [ "$1" == "GPU" ]; then
    export CUDA_VISIBLE_DEVICES=0
    dev=0
else
    usage
    fail "Unknown device $1"
fi

pushd $GCN_EXAMPLE_DIR> /dev/null

# test CPU
python3 gcn.py --dataset cora --gpu $dev || fail "run gcn.py on $1"
python3 gcn_spmv.py --dataset cora --gpu $dev || fail "run gcn_spmv.py on $1"

popd > /dev/null
