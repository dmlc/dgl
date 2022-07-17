CUDA_VISIBLE_DEVICES=6,7 python src/evaluate_model.py \
--dataset Inaturalist \
--bs 384 \
--source_path ~/code/Smooth_AP/data/ --embed_dim 128 \
--resume "INATURALIST_RESNET50_2021-9-2-4-21-21/checkpoint.pth.tar" \
--class_num 948 --loss smoothap \
--trainset lin_train_set1.txt \
--testset Inaturalist_test_set1.txt \
--linsize 29011 --uinsize 18403