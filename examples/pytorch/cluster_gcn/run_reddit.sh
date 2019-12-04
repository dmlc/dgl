#!/bin/bash


python cluster_gcn.py --gpu 0 --dataset reddit-self-loop --lr 1e-2 --weight-decay 0.0 --psize 1500 --batch-size 20 \
  --n-epochs 30 --n-hidden 128 --n-layers 1 --log-every 100 --use-pp --self-loop \
  --note self-loop-reddit-non-sym-ly3-pp-cluster-2-2-wd-5e-4 --dropout 0.2 --use-val --normalize
