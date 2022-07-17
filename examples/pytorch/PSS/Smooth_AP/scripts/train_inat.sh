CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 python src/main.py \
--dataset Inaturalist --lr 1e-5 --fc_lr_mul 1 \
--n_epochs 400 --bs 384 \
--source_path ~/code/Smooth_AP/data/ --embed_dim 128 \
--class_num 948 --loss smoothap --infrequent_eval 1 \
--trainset lin_train_set1.txt --testset Inaturalist_test_set1.txt