#!/bin/bash

# input arguments
DATA="${1-DD}"  # ENZYMES, DD, PROTEINS, COLLAB, IMDB-BINARY, IMDB-MULTI
device=${2-0}
num_trials=${3-10}
print_every=${4-10}


# general settings
hidden_gxn=96
k1=0.8
k2=0.7
sortpooling_k=30
hidden_final=128
batch_size=64
dropout=0.5
cross_weight=1.0
fuse_weight=0.9
weight_decay=1e-3

# dataset-specific settings
case ${DATA} in
IMDB-BINARY)
  num_epochs=200
  patience=40
  learning_rate=0.001
  sortpooling_k=31
  k1=0.8
  k2=0.5
  ;;
IMDB-MULTI)
  num_epochs=200
  patience=40
  learning_rate=0.001
  sortpooling_k=22
  k1=0.8
  k2=0.7
  ;;
COLLAB)
  num_epochs=100
  patience=20
  learning_rate=0.001
  sortpooling_k=130
  k1=0.9
  k2=0.5
  ;;
DD)
  num_epochs=100
  patience=20
  learning_rate=0.0005
  sortpooling_k=291
  k1=0.8
  k2=0.6
  ;;
PROTEINS)
  num_epochs=100
  patience=20
  learning_rate=0.001
  sortpooling_k=32
  k1=0.8
  k2=0.7
  ;;
ENZYMES)
  num_epochs=500
  patience=100
  learning_rate=0.0001
  sortpooling_k=42
  k1=0.7
  k2=0.5
  ;;
*)
  num_epochs=500
  patience=100
  learning_rate=0.00001
  ;;
esac


python main_early_stop.py \
      --dataset $DATA \
      --lr $learning_rate \
      --epochs $num_epochs \
      --hidden_dim $hidden_gxn \
      --final_dense_hidden_dim $hidden_final \
      --readout_nodes $sortpooling_k \
      --pool_ratios $k1 $k2 \
      --batch_size $batch_size \
      --device $device \
      --dropout $dropout \
      --cross_weight $cross_weight\
      --fuse_weight $fuse_weight\
      --weight_decay $weight_decay\
      --num_trials $num_trials\
      --print_every $print_every\
      --patience $patience\
