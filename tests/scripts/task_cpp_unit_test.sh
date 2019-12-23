#!/bin/bash
echo $PWD
pushd build
ls -lh
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
./runUnitTests || fail "CPP unit test"
popd
