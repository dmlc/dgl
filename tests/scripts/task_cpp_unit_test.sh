#!/bin/bash
echo $PWD
pushd build
ls -lh
./runUnitTests || fail "CPP unit test"
popd
