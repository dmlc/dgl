#!/bin/bash
set -e

. /opt/conda/etc/profile.d/conda.sh

conda activate base
pip install --upgrade pip
pip install asv numpy

pushd /root/asv
cat asv.conf.json
asv machine --yes
asv run --verbose
asv publish
popd
