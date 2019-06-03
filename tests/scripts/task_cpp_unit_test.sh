#!/bin/bash
echo $PWD
ls -lh
pushd build
./runUnitTests
popd
