#!/bin/bash

if [ $# -eq 2 ]; then
    MACHINE=$1
    DEVICE=$2
else
    echo "publish.sh <machine_name> <device>"
    exit 1
fi

mkdir -p /tmp/asv_env  # for cached build

docker run --name dgl-reg                   \
           --rm --runtime=nvidia            \
           -v $PWD/..:/asv/dgl              \
           --hostname=$MACHINE -dit dgllib/dgl-ci-gpu:conda /bin/bash
docker exec dgl-reg bash /asv/dgl/benchmarks/run.sh $DEVICE
docker stop dgl-reg
