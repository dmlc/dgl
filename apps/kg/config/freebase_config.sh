# ComplEx CPU
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 500.0 --lr 0.1 --max_step 50000 \
    --batch_size_eval 128 --test -adv --eval_interval 300000 --num_thread 1 \
    --neg_sample_size_eval 100000 --eval_percent 0.02 --num_proc 48

# TransE_l2 8 GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 10 --lr 0.1 --batch_size_eval 1000 \
    --valid --test -adv --mix_cpu_gpu --neg_deg_sample_eval --neg_sample_size_eval 1000 \
    --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --num_thread 4 --regularization_coef 1e-9 \
    --no_eval_filter --max_step 400000 --rel_part --eval_interval 100000 --log_interval 10000 \
    --no_eval_filter --async_update --neg_deg_sample --force_sync_interval 1000

# TransE_l2 16 GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 10 --lr 0.1 --batch_size_eval 1000 \
    --valid --test -adv --mix_cpu_gpu --neg_deg_sample_eval --neg_sample_size_eval 1000 \
    --num_proc 16 --gpu 0 1 2 3 4 5 6 7 --num_thread 4 --regularization_coef 1e-9 \
    --no_eval_filter --max_step 200000 --soft_rel_part --eval_interval 100000 --log_interval 10000 \
    --no_eval_filter --async_update --neg_deg_sample --force_sync_interval 1000