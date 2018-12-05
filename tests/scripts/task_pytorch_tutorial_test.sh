#!/bin/bash
# The working directory for this script will be "tests/scripts"

TUTORIAL_ROOT="../../tutorials"

function fail {
    echo FAIL: $@
    exit -1
}

pushd ${TUTORIAL_ROOT} > /dev/null
# Install requirements
pip3 install -r requirements.txt || fail "installing requirements"

# Test
export MPLBACKEND=Agg
for f in $(find . -name "*.py" ! -name "*_mx.py")
do
    echo "Running tutorial ${f} ..."
    python3 $f || fail "run ${f}"
done

popd > /dev/null
