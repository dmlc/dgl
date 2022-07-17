CUDA_VISIBLE_DEVICES=6,7 python src/evaluate_model.py \
--dataset Inaturalist \
--bs 384 \
--arch vit_base --patch_size 16 \
--source_path ~/code/Smooth_AP/data/ --embed_dim 128 \
--resume "INATURALIST_VIT_BASE_2022-3-6-15-1-42/checkpoint.pth.tar" \
--class_num 948 --loss smoothap \
--trainset lin_train_set1.txt \
--testset Inaturalist_test_set1.txt \
--linsize 29011 --uinsize 18403