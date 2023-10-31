#!/bin/bash

python3 /home/ubuntu/workspace/dgl_2/tools/launch.py \
    --workspace /home/ubuntu/workspace/dgl_2/examples/distributed/graphsage/ \
    --num_trainers 4 \
    --num_servers 2 \
    --num_samplers 0 \
    --part_config /home/ubuntu/workspace/dgl_2/homo_data/ogbn-products.json \
    --ip_config /home/ubuntu/workspace/ip_config.txt \
    "DGL_LIBRARY_PATH=/home/ubuntu/workspace/dgl_2/build PYTHONPATH=tests:/home/ubuntu/workspace/dgl_2/python:tests/python/pytorch/graphbolt:$PYTHONPATH python3 node_classification.py --graph_name ogbn-products --ip_config /home/ubuntu/workspace/ip_config.txt --num_epochs 3 --eval_every 2"
