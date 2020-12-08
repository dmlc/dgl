#!/bin/bash

if [ $# -ne 3 ]; then
    echo "publish.sh <repo> <branch> <machine_name>"
    exit 1
else
    REPO=$1
    BRANCH=$2
    MACHINE=$3
fi

mkdir -p /tmp/asv_env  # for cached build

docker run --name dgl-reg                   \
           --rm --runtime=nvidia            \
           --hostname=$MACHINE -dit dgllib/dgl-ci-gpu:conda /bin/bash
docker exec dgl-reg git clone --recursive https://github.com/$REPO/dgl.git -b $BRANCH /asv/dgl
docker exec dgl-reg bash /asv/dgl/benchmarks/run.sh
docker stop dgl-reg
