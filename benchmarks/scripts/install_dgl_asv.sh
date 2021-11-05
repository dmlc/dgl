#!/bin/bash

set -e

# . /opt/conda/etc/profile.d/conda.sh

DEVICE=${DGL_BENCH_DEVICE:-cpu}

# install
pushd python
rm -rf build *.egg-info dist
pip uninstall -y dgl

if [[ $DEVICE == "cpu" ]]; then
    python3 -m pip install dgl -f https://data.dgl.ai/wheels-internal/repo.html
else
    python3 -m pip install dgl-cu111 -f https://data.dgl.ai/wheels-internal/repo.html
fi

popd