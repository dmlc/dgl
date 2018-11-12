#!/bin/bash
# The working directory for this script will be "tests/scripts"

function fail {
    echo FAIL: $@
    exit -1
}

export MPLBACKEND=Agg

for f in $(find "../../tutorials" -name "*.py")
do
    echo "Running tutorial ${f} ..."
    python3 $f || fail "run ${f}"
done
