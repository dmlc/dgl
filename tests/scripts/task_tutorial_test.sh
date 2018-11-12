#!/bin/bash

function fail {
    echo FAIL: $@
    exit -1
}

export MPLBACKEND=Agg

pushd "tests/scripts" > /dev/null
for f in $(find "../../tutorials" -name "*.py")
do
    echo "Running tutorial ${f} ..."
    python3 $f || fail "run ${f}"
done
popd > /dev/null
