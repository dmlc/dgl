#!/bin/bash
set -e 
echo $PWD
pushd build
ls -lh
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
./runUnitTests
popd
