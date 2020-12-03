#!/bin/bash

set -x
set -e

if [ $# -ne 3 ]; then
    REPO=dmlc
    BRANCH=master
    MACHINE=reg-machine
else
    REPO=$1
    BRANCH=$2
    MACHINE=$3
fi

docker run --name dgl-reg --rm --runtime=nvidia --hostname=reg-machine -dit dgllib/dgl-ci-gpu:conda /bin/bash
docker cp ./asv_data dgl-reg:/root/asv_data/
docker cp ./run.sh dgl-reg:/root/run.sh
docker cp ./requirement.txt dgl-reg:/root/requirement.txt
docker exec dgl-reg bash /root/run.sh $REPO $BRANCH $MACHINE
docker cp dgl-reg:/root/regression/dgl/asv/. ./asv_data/
docker stop dgl-reg
