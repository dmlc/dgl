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

if [[ -v DIST_DGL_TEST_SSH_PORT ]]; then
    SSH_PORT_LINE="-p $DIST_DGL_TEST_SSH_PORT";
fi

if [[ -v DIST_DGL_TEST_SSH_KEY ]]; then
    SSH_KEY_LINE="-i $DIST_DGL_TEST_SSH_KEY";
fi

if [[ -v DIST_DGL_TEST_SSH_SETUP ]]; then
    SSH_SETUP_LINE="$DIST_DGL_TEST_SSH_SETUP;";
fi


while IFS= read line
do
    for pkg in 'pytest' 'psutil' 'torch'
    do
        ret_pkg=$(ssh -o StrictHostKeyChecking=no ${line} ${SSH_PORT_LINE} ${SSH_KEY_LINE} "${SSH_SETUP_LINE}python3 -m pip list | grep -i ${pkg} ") || fail "${pkg} not installed in ${line}"
    done
done < ${DIST_DGL_TEST_IP_CONFIG}

python3 -m pytest -v --capture=tee-sys --junitxml=pytest_dist.xml --durations=100 tests/dist/test_*.py || fail "dist across machines"
