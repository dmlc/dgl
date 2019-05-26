#!/bin/bash
set -e

if [ -d build ]; then
        rm -rf build
fi
mkdir build

rm -rf _download

pushd build
cmake .. -DBUILD_CPP_TEST=1
make -j4
./runUnitTests



