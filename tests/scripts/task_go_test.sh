#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh

function fail {
    echo FAIL: $@
    exit -1
}

export DGLBACKEND=pytorch
export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=tests:${PWD}/python:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}

conda activate pytorch-ci

python3 -m pytest -v --junitxml=pytest_go.xml tests/go || fail "go"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export DMLC_LOG_DEBUG=1
