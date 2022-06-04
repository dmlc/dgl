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

pushd dglgo
rm -rf build *.egg-info dist
pip uninstall -y dglgo
python3 setup.py install
popd

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

python3 -m pytest -v --junitxml=pytest_go.xml tests/go || fail "go"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export DMLC_LOG_DEBUG=1
