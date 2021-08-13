#!/bin/bash

nohup python -u _train_sampling.py --gpu 0 --dataset ppi --sampler node --node-budget 6000 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 >logs/test_lt/ppi/ppi_n_lt 2>&1 &