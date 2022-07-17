CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/get_features.py \
--dataset Inaturalist --lr 1e-5 --fc_lr_mul 0 \
--n_epochs 400 --bs 384 \
--arch vit_base --patch_size 16 \
--source_path ~/code/Smooth_AP/data/ --embed_dim 768 \
--resume INATURALIST_VIT_BASE_2022-3-9-4-46-43/checkpoint.pth.tar \
--finetune false --onehead ture --get_features true --iter 0 \
--class_num 948 --loss smoothap \
--trainset lin_train_set1.txt \
--all_trainset train_set1.txt \
--testset test_set1.txt \
--linsize 29011 --uinsize 18403 \
--cluster_path '/home/ubuntu/code/hilander/hilander/data/inat_hilander_l_smoothap_train_selectbydensity_expand_0.8_0.9_iter1_vit.pkl'
