#!/bin/bash

KG_DIR="./apps/kg/"

function fail {
    echo FAIL: $@
    exit -1
}

function usage {
    echo "Usage: $0 [cpu|gpu]"
}

# check arguments
if [ $# -ne 1 ]; then
    usage
    fail "Error: must specify device"
fi

if [ "$1" == "cpu" ]; then
    dev=-1
elif [ "$1" == "gpu" ]; then
    export CUDA_VISIBLE_DEVICES=0
    dev=0
else
    usage
    fail "Unknown device $1"
fi

export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=${PWD}/python:$KG_DIR:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}

# test

pushd $KG_DIR> /dev/null

python3 tests/test_score.py || "run test_score.py on $1"

popd > /dev/null
