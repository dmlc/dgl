#!/bin/bash
set -e

if [ $# -ne 3 ]; then
    echo "run.sh <repo> <branch>"
    exit 1
fi

REPO=$1
BRANCH=$2

. /opt/conda/etc/profile.d/conda.sh

# install dgl

cd ~
git clone --recursive https://github.com/$REPO/dgl.git 
cd dgl
git checkout $BRANCH

conda activate base
pip install --upgrade pip
pip install asv numpy

conda activate base
asv machine --yes
asv run
asv publish
