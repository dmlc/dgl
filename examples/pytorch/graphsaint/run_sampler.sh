#!/bin/zsh

_python=/home/ubuntu/anaconda3/envs/pytorch_latest_p37/bin/python3.7
_file=/home/ubuntu/dgl/examples/pytorch/graphsaint/train_sampling.py
#_args="--gpu 0 --dataset flickr --sampler node --node-budget 8000 --num-repeat 25 --n-epochs 30 --n-hidden 256 --arch 1-1-0 --dropout 0.2"
_log_dir=/home/ubuntu/dgl/examples/pytorch/graphsaint/logs
_task=$1
_log_file=$2

#$_python $_file "$_args"
nohup $_python -u $_file --task $_task >$_log_dir/$_log_file 2>&1 &
#echo $_python
#echo $_file
#echo $_args
