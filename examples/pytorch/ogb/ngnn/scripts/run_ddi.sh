python main.py \
    --dataset ogbl-ddi \
    --device 3 \
    --ngnn_type input \
    --use_sage \
    --epochs 600 \
    --dropout 0.25 \
    --num_layers 2 \
    --lr 0.0012 \
    --batch_size 32768 \
    --runs 50 \
    | tee results/ddi-input-SAGE-epoch_600-dropout_0.25-layers_2-lr_0.0012-batch_32768
