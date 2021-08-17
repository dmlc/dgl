#!/bin/bash

_dataset='ppi'
_sampler='n'
_task="${_dataset}_${_sampler}"
_log="n_on"
nohup python -u train_sampling.py --task $_task >logs/final2/$_dataset/$_log 2>&1 &

