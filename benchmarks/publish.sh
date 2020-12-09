#!/bin/bash

if [ $# -eq 2 ]; then
    MACHINE=$1
    DEVICE=$2
else
    echo "publish.sh <machine_name> <device>"
    exit 1
fi

WS_ROOT=/asv/dgl

docker run --name dgl-reg                   \
           --rm --runtime=nvidia            \
           --hostname=$MACHINE -dit dgllib/dgl-ci-gpu:conda /bin/bash
docker exec dgl-reg mkdir -p $WS_ROOT
docker cp ../.git dgl-reg:$WS_ROOT
docker cp . dgl-reg:$WS_ROOT/benchmarks/
docker exec dgl-reg bash $WS_ROOT/benchmarks/run.sh $DEVICE
docker cp dgl-reg:$WS_ROOT/benchmarks/results results
docker cp dgl-reg:$WS_ROOT/benchmarks/html html
docker stop dgl-reg
