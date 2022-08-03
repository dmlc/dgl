python dgl__gnn.py \
    --device 1 \
    --ngnn_type input \
    --use_sage \
    --epochs 50 \
    --dropout 0.25 \
    --num_layers 3 \
    --lr 0.001 \
    --batch_size 65536 \
    --runs 10 \
    | tee ppa-input-SAGE-epoch_50-dropout_0.25-layers_3-lr_0.001-batch_65536