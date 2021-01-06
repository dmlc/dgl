#!/bin/bash

# The script launches a docker container to run ASV benchmarks. We use the same docker
# image as our CI (i.e., dgllib/dgl-ci-gpu:conda). It performs the following steps:
#
#   1. Start a docker container of the given machine name. The machine name will be
#      displayed on the generated website.
#   2. Copy `.git` into the container. It allows ASV to determine the repository information
#      such as commit hash, branches, etc.
#   3. Copy this folder into the container including the ASV configuration file `asv.conf.json`.
#      This means any changes to the files in this folder do not
#      require a git commit. By contrast, to correctly benchmark your changes to the core
#      library (e.g., "python/dgl"), you must call git commit first.
#   4. It then calls the `run.sh` script inside the container. It will invoke `asv run`.
#      You can change the command such as specifying the benchmarks to run or adding some flags.
#   5. After benchmarking, it copies the generated `results` and `html` folders back to
#      the host machine.
#

if [ $# -eq 2 ]; then
    MACHINE=$1
    DEVICE=$2
else
    echo "publish.sh <machine_name> <device>"
    exit 1
fi

WS_ROOT=/asv/dgl

docker pull dgllib/dgl-ci-gpu:conda

echo $HOME

if [[ $DEVICE == "cpu" ]]; then
    docker run --name dgl-reg \
        --rm \
        -v /mnt/efs/fs1/:/tmp/dataset \
        --hostname=$MACHINE -dit dgllib/dgl-ci-gpu:conda /bin/bash
else
    docker run --name dgl-reg \
        --rm --runtime=nvidia \
        -v /mnt/efs/fs1/:/tmp/dataset \
        --hostname=$MACHINE -dit dgllib/dgl-ci-gpu:conda /bin/bash
fi

docker exec dgl-reg mkdir -p $WS_ROOT
docker cp ../../.git dgl-reg:$WS_ROOT
docker cp ../ dgl-reg:$WS_ROOT/benchmarks/
docker cp torch_gpu_pip.txt dgl-reg:/asv
docker exec dgl-reg bash $WS_ROOT/benchmarks/run.sh $DEVICE
docker cp dgl-reg:$WS_ROOT/benchmarks/results ../
docker cp dgl-reg:$WS_ROOT/benchmarks/html ../
docker stop dgl-reg
