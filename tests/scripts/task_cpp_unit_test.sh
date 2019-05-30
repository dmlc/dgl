#!/bin/bash
echo $PWD
ls -lh
ls .. -lh
pushd build
./runUnitTests
popd
