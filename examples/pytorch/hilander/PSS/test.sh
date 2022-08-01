python Smooth_AP/src/evaluate_model.py \
--dataset Inaturalist \
--bs 384 \
--source_path ~/code/Smooth_AP/data/ --embed_dim 128 \
--resume $CHECKPOINT_PATH \
--class_num 948 --loss smoothap \
--trainset lin_train_set1.txt \
--testset Inaturalist_test_set1.txt \
--linsize 29011 --uinsize 18403