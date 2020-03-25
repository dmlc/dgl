#!/bin/bash

set -x

if [ $# -ne 2 ]; then
    REPO=dmlc
    BRANCH=master
else
    REPO=$1
    BRANCH=$2
fi

docker run --name dgl-reg --rm --hostname=reg-machine --runtime=nvidia -dit dgllib/dgl-ci-gpu:conda /bin/bash
docker cp /home/ubuntu/asv_data dgl-reg:/root/asv_data/
docker exec dgl-reg bash /root/asv_data/run.sh $REPO $BRANCH
docker cp dgl-reg:/root/regression/dgl/asv/. /home/ubuntu/asv_data/
docker stop dgl-reg
