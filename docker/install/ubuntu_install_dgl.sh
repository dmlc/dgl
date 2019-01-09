#!/bin/bash
git clone --recursive https://github.com/dmlc/dgl.git
cd dgl
cp ./tests/scripts/build_dgl.sh ./
bash build_dgl.sh
exec "$@"

