#!/bin/bash
function fail {
    echo FAIL: $@
    exit -1
}

echo $PWD
export DGLBACKEND=pytorch
export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=${PWD}/tests:${PWD}/python:$PYTHONPATH
export LD_LIBRARY_PATH=${PWD}/build:$LD_LIBRARY_PATH
export DIST_DGL_TEST_CPP_BIN_DIR=${PWD}/build
export DIST_DGL_TEST_IP_CONFIG=/home/ubuntu/workspace/ip_config.txt
export DIST_DGL_TEST_PY_BIN_DIR=${PWD}/tests/dist/python

while IFS= read line
do
    for pkg in 'pytest' 'psutil' 'torch'
    do
        ret_pkg=$(ssh ${line} "python3 -m pip list | grep -i ${pkg} ") || fail "${pkg} not installed in ${line}"
    done
done < ${DIST_DGL_TEST_IP_CONFIG}

python3 -m pytest -v --capture=tee-sys --junitxml=pytest_dist.xml tests/dist/test_*.py || fail "dist across machines"
