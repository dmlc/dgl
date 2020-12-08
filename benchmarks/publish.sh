#!/bin/bash

if [ $# -ne 1 ]; then
    echo "publish.sh <machine_name>"
    exit 1
else
    MACHINE=$1
fi

mkdir -p /tmp/asv_env  # for cached build

docker run --name dgl-reg                   \
           --rm --runtime=nvidia            \
           --hostname=$MACHINE -dit dgllib/dgl-ci-gpu:conda /bin/bash
docker cp .. dgl-reg:/root/
docker exec dgl-reg bash /root/dgl/benchmarks/run.sh
docker stop dgl-reg
