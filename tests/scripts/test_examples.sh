#!/bin/bash

GCN_EXAMPLE_DIR="../../examples/pytorch/gcn"

# test with one GPU
export CUDA_VISIBLE_DEVICES=0

function fail {
    echo FAIL: $@
    exit -1
}

pushd $GCN_EXAMPLE_DIR> /dev/null

# test CPU
python gcn.py --dataset cora --gpu -1 || fail "run gcn.py CPU"
python gcn_spmv.py --dataset cora --gpu -1 || fail "run gcn_spmv.py on CPU"

# test GPU
python gcn.py --dataset cora --gpu 0 || fail "run gcn.py GPU"
python gcn_spmv.py --dataset cora --gpu 0 || fail "run gcn_spmv.py GPU"

popd > /dev/null
