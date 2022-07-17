CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0 python src/finetune_1head.py \
--dataset Inaturalist --lr 1e-5 --fc_lr_mul 1 \
--n_epochs 400 --bs 384 --class_num 1024 \
--source_path ~/code/Smooth_AP/data/ --embed_dim 128 \
--trainset lin_train_set1.txt --testset Inaturalist_test_set1.txt \
--cluster_path '/home/ubuntu/code/hilander/hilander/data/inat_hilander_l_smoothap_train_selectbydensity_0.8_0.9_iter0.pkl' \
--finetune true --onehead true --loss smoothap --infrequent_eval 1
