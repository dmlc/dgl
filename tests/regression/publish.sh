#!/bin/bash

set -x

if [ $# -ne 3 ]; then
    REPO=dmlc
    BRANCH=master
    MACHINE=reg-machine
else
    REPO=$1
    BRANCH=$2
    MACHINE=$3
fi

mkdir -p asv_data

docker run --name dgl-reg \
           --rm --runtime=nvidia \
           -v asv_data:/asv \
           --hostname=$MACHINE -dit dgllib/dgl-ci-gpu:conda /bin/bash
docker cp ./run.sh dgl-reg:/root/run.sh
docker exec dgl-reg bash /root/run.sh $REPO $BRANCH
docker stop dgl-reg
