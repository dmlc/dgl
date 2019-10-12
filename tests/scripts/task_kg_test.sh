#!/bin/bash

KG_DIR="./apps/kg/"

function fail {
    echo FAIL: $@
    exit -1
}

function usage {
    echo "Usage: $0 backend device"
}

# check arguments
if [ $# -ne 2 ]; then
    usage
    fail "Error: must specify device and bakend"
fi

if [ "$2" == "cpu" ]; then
    dev=-1
elif [ "$2" == "gpu" ]; then
    export CUDA_VISIBLE_DEVICES=0
    dev=0
else
    usage
    fail "Unknown device $2"
fi

export DGLBACKEND=$1
export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=${PWD}/python:$KG_DIR:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}

# test

pushd $KG_DIR> /dev/null

python3 -m nose -v --with-xunit tests/test_score.py || fail "run test_score.py on $1"

if [ "$2" == "cpu" ]; then
    # verify CPU training DistMult
    python3 train.py --model DistMult --dataset FB15k --batch_size 128 \
        --neg_sample_size 16 --hidden_dim 100 --gamma 500.0 --lr 0.1 --max_step 100 \
        --batch_size_eval 16 --valid --test -adv --eval_interval 30 --eval_percent 0.01 \
        --save_emb DistMult_FB15k_emb --data_path /data/kg || fail "run DistMult on $2"

    # verify saving training result
    python3 eval.py --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 500.0 --batch_size 16 --model_path DistMult_FB15k_emb/ \
        --eval_percent 0.01 --data_path /data/kg || fail "eval DistMult on $2"
elif [ "$2" == "gpu" ]; then
    # verify GPU training DistMult
    python3 train.py --model DistMult --dataset FB15k --batch_size 128 \
        --neg_sample_size 16 --hidden_dim 100 --gamma 500.0 --lr 0.1 --max_step 100 \
        --batch_size_eval 16 --gpu 0 --valid --test -adv --eval_interval 30 --eval_percent 0.01 \
        --data_path /data/kg || fail "run DistMult on $2"

    # verify mixed CPU GPU training
    python3 train.py --model DistMult --dataset FB15k --batch_size 128 \
        --neg_sample_size 16 --hidden_dim 100 --gamma 500.0 --lr 0.1 --max_step 100 \
        --batch_size_eval 16 --gpu 0 --valid --test -adv --mix_cpu_gpu --eval_percent 0.01 \
        --save_emb DistMult_FB15k_emb --data_path /data/kg || fail "run mix CPU/GPU DistMult"

    # verify saving training result
    python3 eval.py --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 500.0 --batch_size 16 --gpu 0 --model_path DistMult_FB15k_emb/ \
        --eval_percent 0.01 --data_path /data/kg || fail "eval DistMult on $2"
fi

popd > /dev/null
