#!/bin/bash

GCN_EXAMPLE_DIR="../../examples/pytorch/gcn"

function fail {
    echo FAIL: $@
    exit -1
}

pushd $GCN_EXAMPLE_DIR> /dev/null

# test CPU
python gcn.py --dataset cora --gpu -1 || fail "run gcn.py CPU"
python gcn_spmv.py --dataset cora --gpu -1 || fail "run gcn_spmv.py on CPU"

popd > /dev/null
