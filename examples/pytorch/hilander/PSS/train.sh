#!/bin/bash

mkdir hilander_checkpoint

####################### ITER 0 #######################
# iter 0 (supervised baseline) - train Smooth-AP
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python Smooth_AP/src/main.py \
--dataset Inaturalist --lr 1e-5 --fc_lr_mul 1 \
--n_epochs 400 --bs 384 \
--source_path "../../data/" --embed_dim 128 \
--class_num 948 --loss smoothap --infrequent_eval 1 \
--trainset lin_train_set1.txt --testset Inaturalist_test_set1.txt

# iter 0 (supervised baseline) - get feature
python Smooth_AP/src/get_features.py \
--dataset Inaturalist --lr 1e-5 --fc_lr_mul 1 \
--n_epochs 400 --bs 384 \
--source_path "../../data/" --embed_dim 128 \
--resume "0/checkpoint_0.pth.tar" \
--finetune false --get_features true --iter 0 \
--class_num 948 --loss smoothap \
--trainset lin_train_set1.txt \
--all_trainset train_set1.txt \
--testset Inaturalist_test_set1.txt \
--linsize 29011

# iter 0 (supervised baseline) - train hi-lander
python train_subg_inat.py \
--data_path "/home/ubuntu/code/dgl/examples/pytorch/hilander/PSS/data/Inaturalist/T_train_iter0_smoothap_inat_features.pkl" \
--model_filename '/home/ubuntu/code/dgl/examples/pytorch/hilander/PSS/hilander_checkpoint/inat_l_smoothap_iter0.pth' \
--knn_k 10,5,3 --levels 2,3,4 \
--hidden 512 --epochs 1000 --lr 0.01 \
--batch_size 4096 --num_conv 1 --gat --balance

# iter 0 (supervised baseline) - get pseudo labels
python test_subg_inat.py \
--data_path '/home/ubuntu/code/dgl/examples/pytorch/hilander/PSS/data/Inaturalist/all_train_iter0_smoothap_inat_features.pkl' \
--model_filename '/home/ubuntu/code/dgl/examples/pytorch/hilander/PSS/hilander_checkpoint/inat_l_smoothap_iter0.pth'  --knn_k 10 \
--tau 0.9 --level 10 --threshold prob \
--hidden 512 --num_conv 1 --gat --batch_size 4096 --early_stop \
--mode selectbydensity --thresh 0.8 \
--linsize 29011 --uinsize 18403 --inclasses 948 \
--output_filename 'data/inat_hilander_l_smoothap_train_selectbydensity_iter0.pkl'


for i in {1..4} ; do
  last_iter=`expr $i - 1`
  echo ${last_iter}
  # iter i - train Smooth-AP
  python Smooth_AP/src/finetune_1head.py \
  --dataset Inaturalist --lr 1e-5 --fc_lr_mul 1 \
  --n_epochs 400 --bs 384 --class_num 1024 \
  --source_path "../../data/" --embed_dim 128 \
  --trainset lin_train_set1.txt --testset Inaturalist_test_set1.txt \
  --cluster_path "../../data/inat_hilander_l_smoothap_train_selectbydensity_iter${last_iter}.pkl" \
  --finetune true --loss smoothap --infrequent_eval 1 --iter ${i}

  # iter i - get feature
  python Smooth_AP/src/get_features.py \
  --dataset Inaturalist --lr 1e-5 --fc_lr_mul 1 \
  --n_epochs 400 --bs 384 \
  --source_path "../../data/" --embed_dim 128 \
  --resume "${i}/checkpoint_${i}.pth.tar" \
  --finetune false --get_features true --iter ${i} \
  --class_num 948 --loss smoothap \
  --trainset lin_train_set1.txt \
  --all_trainset train_set1.txt \
  --testset Inaturalist_test_set1.txt \
  --linsize 29011 --uinsize 18403 \
  --cluster_path "../../data/inat_hilander_l_smoothap_train_selectbydensity_iter${last_iter}.pkl"

  # iter i - train hi-lander
  python train_subg_inat.py \
  --data_path "/home/ubuntu/code/dgl/examples/pytorch/hilander/PSS/data/Inaturalist/T_train_iter${i}_smoothap_inat_features.pkl" \
  --model_filename "/home/ubuntu/code/dgl/examples/pytorch/hilander/PSS/hilander_checkpoint/inat_l_smoothap_iter${i}.pth" \
  --knn_k 10,5,3 --levels 2,3,4 \
  --hidden 512 --epochs 1000 --lr 0.01 \
  --batch_size 4096 --num_conv 1 --gat --balance

  # iter i - get pseudo labels
  python test_subg_inat.py \
  --data_path "/home/ubuntu/code/dgl/examples/pytorch/hilander/PSS/data/Inaturalist/all_train_iter${i}_smoothap_inat_features.pkl" \
  --model_filename "/home/ubuntu/code/dgl/examples/pytorch/hilander/PSS/hilander_checkpoint/inat_l_smoothap_iter${i}.pth"  --knn_k 10 \
  --tau 0.9 --level 10 --threshold prob \
  --hidden 512 --num_conv 1 --gat --batch_size 4096 --early_stop \
  --mode selectbydensity --thresh 0.8 \
  --linsize 29011 --uinsize 18403 --inclasses 948 \
  --output_filename "data/inat_hilander_l_smoothap_train_selectbydensity_iter${i}.pkl"
done
