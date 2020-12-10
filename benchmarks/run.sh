#!/bin/bash

set -e

DEVICE=$1
ROOT=/asv/dgl

. /opt/conda/etc/profile.d/conda.sh

conda activate base
pip install --upgrade pip
pip install asv
pip uninstall -y dgl

export DGL_BENCH_DEVICE=$DEVICE
pushd $ROOT/benchmarks
cat asv.conf.json
asv machine --yes
asv run -e --verbose
asv publish
popd
