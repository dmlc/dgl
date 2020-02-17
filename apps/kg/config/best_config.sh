#To reproduce reported results on README, you can run the model with the following commands:

# for FB15k
# DistMult 1GPU
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 \
    --valid --test -adv --mix_cpu_gpu --eval_interval 100000 --gpu 0 \
    --num_worker=8  --max_step 40000
# DistMult 8GPU
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 \
    --valid --test -adv --mix_cpu_gpu --eval_interval 100000 --num_proc 8 --gpu 0 1 2 3 4 5 6 7 \
    --num_worker=4 --max_step 10000 --rel_part --async_update

# ComplEx 1GPU
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.1 --regularization_coef 2.00E-06 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 \
    --gpu 0 --num_worker=8 --max_step 32000
# ComplEx 8GPU
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.1 --regularization_coef 2.00E-06 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --num_worker=4 --max_step 4000 --rel_part --async_update

# TransE_l1 1GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l1 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 64 --regularization_coef 1e-07 --hidden_dim 400 --gamma 16.0 --lr 0.01 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 \
    --gpu 0 --num_worker=8 --max_step 48000
# TransE_l1 8GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l1 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 64 --regularization_coef 1e-07 --hidden_dim 400 --gamma 16.0 --lr 0.01 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --num_worker=4 --max_step 6000 --rel_part --async_update

# TransE_l2 1GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 12.0 --lr 0.1 --max_step 30000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv --regularization_coef=2e-7 

# RESCAL 1GPU
DGLBACKEND=pytorch python3 train.py --model RESCAL --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 500 --gamma 24.0 --lr 0.03 --max_step 30000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv

# TransR 1GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 5e-8 --hidden_dim 200 --gamma 8.0 --lr 0.015 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 \
    --gpu 0 --num_worker=8 --max_step 32000
# TransR 8GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 5e-8 --hidden_dim 200 --gamma 8.0 --lr 0.015 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --num_worker=4 --max_step 4000 --rel_part --async_update

# RotatE 1GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset FB15k --batch_size 2048 \
    --neg_sample_size 256 --regularization_coef 1e-07 --hidden_dim 200 --gamma 12.0 --lr 0.009 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 -de \
    --mix_cpu_gpu --max_step 40000 --gpu 0 --num_worker=4

# RotatE 8GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset FB15k --batch_size 2048 \
    --neg_sample_size 256 --regularization_coef 1e-07 --hidden_dim 200 --gamma 12.0 --lr 0.009 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 -de \
    --mix_cpu_gpu --max_step 5000 --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --num_worker=4 \
    --rel_part --async_update

# for wn18
DGLBACKEND=pytorch python3 train.py --model TransE_l1 --dataset wn18 --batch_size 1024 \
    --neg_sample_size 512 --hidden_dim 500 --gamma 12.0 --adversarial_temperature 0.5 \
    --lr 0.01 --max_step 40000 --batch_size_eval 16 --gpu 0 --valid --test -adv \
    --regularization_coef 0.00001

DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset wn18 --batch_size 1024 \
    --neg_sample_size 512 --hidden_dim 500 --gamma 6.0 --lr 0.1 --max_step 20000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv --regularization_coef 0.0000001

DGLBACKEND=pytorch python3 train.py --model DistMult --dataset wn18 --batch_size 1024 \
    --neg_sample_size 1024 --hidden_dim 1000 --gamma 200.0 --lr 0.1 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv --regularization_coef 0.00001

DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset wn18 --batch_size 1024 \
    --neg_sample_size 1024 --hidden_dim 500 --gamma 200.0 --lr 0.1 --max_step 20000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv --regularization_coef 0.00001

DGLBACKEND=pytorch python3 train.py --model RESCAL --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 250 --gamma 24.0 --lr 0.03 --max_step 20000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv

DGLBACKEND=pytorch python3 train.py --model TransR --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 500 --gamma 16.0 --lr 0.1 --max_step 30000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv

DGLBACKEND=pytorch python3 train.py --model RotatE --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 12.0 --lr 0.02 --max_step 20000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv -de

# for Freebase

DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 500.0     --lr 0.1 --max_step 50000 \
    --batch_size_eval 128 --test -adv --eval_interval 300000 \
    --neg_sample_size_test 100000 --eval_percent 0.02 --num_proc 64

# Freebase multi-gpu
# TransE_l2 8 GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 10 --lr 0.1 --batch_size_eval 1000 \
    --valid --test -adv --mix_cpu_gpu --neg_deg_sample_eval --neg_sample_size_test 1000 \
    --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --num_worker 4 --regularization_coef 1e-9 \
    --no_eval_filter --max_step 400000 --rel_part --eval_interval 100000 --log_interval 10000 \
    --no_eval_filter --async_update --neg_deg_sample --force_sync_interval 1000

# TransE_l2 16 GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 10 --lr 0.1 --batch_size_eval 1000 \
    --valid --test -adv --mix_cpu_gpu --neg_deg_sample_eval --neg_sample_size_test 1000 \
    --num_proc 16 --gpu 0 1 2 3 4 5 6 7 --num_worker 4 --regularization_coef 1e-9 \
    --no_eval_filter --max_step 200000 --soft_rel_part --eval_interval 100000 --log_interval 10000 \
    --no_eval_filter --async_update --neg_deg_sample --force_sync_interval 1000
