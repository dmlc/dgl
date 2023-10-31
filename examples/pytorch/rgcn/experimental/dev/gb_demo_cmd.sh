#!/bin/bash

python3 /home/ubuntu/workspace/dgl_2/tools/launch.py \
    --workspace /home/ubuntu/workspace/dgl_2/examples/pytorch/rgcn/experimental/ \
    --num_trainers 4 \
    --num_servers 2 \
    --num_samplers 0 \
    --part_config /home/ubuntu/workspace/dgl_2/data/ogbn-mag.json \
    --ip_config /home/ubuntu/workspace/ip_config.txt \
    "DGL_LIBRARY_PATH=/home/ubuntu/workspace/dgl_2/build PYTHONPATH=tests:/home/ubuntu/workspace/dgl_2/python:tests/python/pytorch/graphbolt:$PYTHONPATH python3 gb_demo.py --graph-name ogbn-mag --dataset ogbn-mag --ip-config /home/ubuntu/workspace/ip_config.txt --fanout='25,10' --batch-size 1024  --n-hidden 64 --lr 0.01 --eval-batch-size 1024 --graphbolt"
