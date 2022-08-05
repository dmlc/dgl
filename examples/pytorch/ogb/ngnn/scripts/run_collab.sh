python main.py \
    --dataset ogbl-collab \
    --device 0 \
    --ngnn_type hidden \
    --use_sage \
    --epochs 600 \
    --dropout 0.2 \
    --num_layers 3 \
    --lr 0.001 \
    --batch_size 32768 \
    --runs 10 \
    | tee results/collab-hidden-SAGE-epoch_600-dropout_0.2-layers_3-lr_0.001-batch_32768