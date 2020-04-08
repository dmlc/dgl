#!/bin/bash
set -e

if [ $# -ne 2 ]; then
    echo "run.sh <repo> <branch>"
    exit 1
fi

REPO=$1
BRANCH=$2

. /opt/conda/etc/profile.d/conda.sh

cd ~
mkdir regression
cd regression
# git config core.filemode false
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
asv machine --yes
asv run
asv publish
