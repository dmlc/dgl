#!/bin/bash
set -e

if [ $# -ne 3 ]; then
    echo "run.sh <repo> <branch> <machine>"
    exit 1
fi

REPO=$1
BRANCH=$2
MACHINE=$3

. /opt/conda/etc/profile.d/conda.sh

cd ~
mkdir regression
cd regression
git clone --recursive https://github.com/$REPO/dgl.git 
cd dgl
git checkout $BRANCH
mkdir asv
cp -r ~/asv_data/* asv/

conda activate base
pip install --upgrade pip
pip install asv numpy

export DGL_LIBRARY_PATH="~/dgl/build"

conda activate base
asv machine --machine $MACHINE --yes
asv run --environment base
asv publish
