#!/bin/bash

set -e

DEVICE=$1
ROOT=/asv/dgl

. /opt/conda/etc/profile.d/conda.sh

conda activate base
pip install --upgrade pip
# Newer asv version like 0.5.1 has different result format,
# so we fix the version here. Or `generate_excel.py` has to be changed.
pip install asv==0.4.2
pip uninstall -y dgl

export DGL_BENCH_DEVICE=$DEVICE
echo "DGL_BENCH_DEVICE=$DGL_BENCH_DEVICE"
pushd $ROOT/benchmarks
cat asv.conf.json
asv machine --yes
# If --launch-method is specified as 'spawn', multigpu tests will crash with
# "No module named 'benchmarks' is found".
asv run -e -v
asv publish
popd
