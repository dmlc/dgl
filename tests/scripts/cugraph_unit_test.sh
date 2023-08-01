#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh

function fail {
    echo FAIL: $@
    exit -1
}

export DGLBACKEND=$1
export DGLTESTDEV=gpu
export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=tests:${PWD}/python:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}/_download
export TF_FORCE_GPU_ALLOW_GROWTH=true

export CUDA_VISIBLE_DEVICES=0

python3 -m pip install pytest psutil pyyaml pydantic pandas rdflib ogb || fail "pip install"

python3 -m pytest -v --junitxml=pytest_cugraph.xml --durations=20 tests/cugraph || fail "cugraph"
