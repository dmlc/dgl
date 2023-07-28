#!/bin/bash
# The working directory for this script will be "tests/scripts"

. /opt/conda/etc/profile.d/conda.sh
conda activate pytorch-ci
TUTORIAL_ROOT="./tutorials"

function fail {
    echo FAIL: $@
    exit -1
}

export MPLBACKEND=Agg
export DGLBACKEND=pytorch
export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=${PWD}/python:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}/_download

pushd ${TUTORIAL_ROOT} > /dev/null
# Install requirements
pip install -r requirements.txt || fail "installing requirements"

# Test
for f in $(find . -path ./dist -prune -false -o -name "*.py" ! -name "*_mx.py")
do
    echo "Running tutorial ${f} ..."
    python3 $f || fail "run ${f}"
done

popd > /dev/null
