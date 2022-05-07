#!/bin/bash
function fail {
    echo FAIL: $@
    exit -1
}

echo $PWD
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
export DIST_DGL_TEST_CPP_BIN_DIR=$PWD/build
export DIST_DGL_TEST_IP_CONFIG=/home/ubuntu/workspace/ip_config.txt

python3 -m pip install pytest || fail "pip install"
python3 -m pytest -v --capture=tee-sys --junitxml=pytest_dist.xml tests/dist/*.py || fail "dist across machines"
