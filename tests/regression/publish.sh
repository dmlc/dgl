#!/bin/bash

if [ $# -ne 1 ]; then
    echo "publish.sh <machine_name>"
    exit 1
else
    MACHINE=$1
fi

mkdir -p /tmp/asv_env  # for cached build

docker run --name dgl-reg          \
           --rm --runtime=nvidia   \
           -v /tmp/asv_env:/asv/env  \
           --hostname=$MACHINE -dit dgllib/dgl-ci-gpu:conda /bin/bash
docker cp ./run.sh dgl-reg:/root/run.sh
docker exec dgl-reg bash /root/run.sh
docker cp dgl-reg:/asv/results asv_data/
docker cp dgl-reg:/asv/html asv_data/
docker stop dgl-reg
