#!/bin/bash
set -e

. /opt/conda/etc/profile.d/conda.sh

pushd /root/dgl

conda activate base
pip install --upgrade pip
pip install asv numpy

conda activate base
cat asv.conf.json
asv machine --yes
asv run --verbose
asv publish

popd
