python main.py \
    --dataset ogbl-ddi \
    --ratio_per_hop 0.2 \
    --use_edge_weight \
    --eval_steps 1 \
    --epochs 10 \
    --dynamic_val \
    --dynamic_test \
    --train_percent 5