CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/main.py \
--dataset Inaturalist --lr 1e-5 --fc_lr_mul 0 \
--arch 'vit_base' --patch_size 16 \
--n_epochs 400 --bs 384 \
--pretrained_weights ~/code/Smooth_AP/data/dino_vitbase16_pretrain.pth \
--source_path ~/code/Smooth_AP/data/ --embed_dim 768 \
--class_num 948 --loss smoothap --infrequent_eval 1 \
--trainset 'lin_train_limitclass.txt' \
--linsize 29011 --uinsize 18403