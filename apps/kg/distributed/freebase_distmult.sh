#!/bin/bash

##################################################################################
# This script runing distmult model on Freebase dataset in distributed setting.
# You can change the hyper-parameter in this file but DO NOT run script manually
##################################################################################
machine_id=$1
server_count=$2
machine_count=$3

# Delete the temp file
rm *-shape

##################################################################################
# Start kvserver
##################################################################################
SERVER_ID_LOW=$((machine_id*server_count))
SERVER_ID_HIGH=$(((machine_id+1)*server_count))

while [ $SERVER_ID_LOW -lt $SERVER_ID_HIGH ]
do
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 ../kvserver.py --model DistMult --dataset Freebase \
    --hidden_dim 400 --gamma 143.0 --lr 0.08 --total_client 160 --server_id $SERVER_ID_LOW &
    let SERVER_ID_LOW+=1
done

##################################################################################
# Start kvclient
##################################################################################
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 ../kvclient.py --model DistMult --dataset Freebase \
--batch_size 1024 --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --max_step 12500 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --total_machine $machine_count --num_thread 1 --num_client 40