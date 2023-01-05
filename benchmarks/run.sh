#!/bin/bash

set -e

DEVICE=$1
ROOT=/asv/dgl

. /opt/conda/etc/profile.d/conda.sh

conda activate base
pip install --upgrade pip
pip install asv==0.4.2
pip uninstall -y dgl

export DGL_BENCH_DEVICE=$DEVICE
echo "DGL_BENCH_DEVICE=$DGL_BENCH_DEVICE"
pushd $ROOT/benchmarks
cat asv.conf.json
asv machine --yes
asv run --launch-method=spawn -e -v
asv publish
ls -lh ${ROOT}/benchmarks/*
popd
