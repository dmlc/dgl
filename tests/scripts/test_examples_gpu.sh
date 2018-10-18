#!/bin/bash

GCN_EXAMPLE_DIR="../../examples/pytorch/gcn"

# test with one GPU
export CUDA_VISIBLE_DEVICES=0

function fail {
    echo FAIL: $@
    exit -1
}

pushd $GCN_EXAMPLE_DIR> /dev/null

# test GPU
python3 gcn.py --dataset cora --gpu 0 || fail "run gcn.py GPU"
python3 gcn_spmv.py --dataset cora --gpu 0 || fail "run gcn_spmv.py GPU"

popd > /dev/null
