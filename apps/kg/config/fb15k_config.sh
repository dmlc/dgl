# DistMult 1GPU
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 \
    --valid --test -adv --mix_cpu_gpu --eval_interval 100000 --gpu 0 --num_thread 4 --max_step 40000

# DistMult 8GPU
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 \
    --valid --test -adv --mix_cpu_gpu --eval_interval 100000 --num_proc 8 --gpu 0 1 2 3 4 5 6 7 \
    --max_step 10000 --num_thread 4 --rel_part --async_update

# ComplEx 1GPU
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.1 --regularization_coef 2.00E-06 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 \
    --gpu 0 --num_thread 4 --max_step 32000

# ComplEx 8GPU
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.1 --regularization_coef 2.00E-06 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --max_step 4000 --num_thread 4 --rel_part --async_update

# TransE_l1 1GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l1 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 64 --regularization_coef 1e-07 --hidden_dim 400 --gamma 16.0 --lr 0.01 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 \
    --gpu 0 --num_thread 4 --max_step 48000

# TransE_l1 8GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l1 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 64 --regularization_coef 1e-07 --hidden_dim 400 --gamma 16.0 --lr 0.01 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --max_step 6000 --num_thread 4 --rel_part --async_update

# TransE_l2 1GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 12.0 --lr 0.1 --max_step 30000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv --num_thread 4 --regularization_coef=2e-7 

# RESCAL 1GPU
DGLBACKEND=pytorch python3 train.py --model RESCAL --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 500 --gamma 24.0 --lr 0.03 --max_step 30000 \
    --batch_size_eval 16 --gpu 0 --num_thread 4 --valid --test -adv

# TransR 1GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 5e-8 --hidden_dim 200 --gamma 8.0 --lr 0.015 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 \
    --gpu 0 --num_thread 4 --max_step 32000
    
# TransR 8GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 5e-8 --hidden_dim 200 --gamma 8.0 --lr 0.015 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --max_step 4000 --num_thread 4 --rel_part --async_update

# RotatE 1GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset FB15k --batch_size 2048 \
    --neg_sample_size 256 --regularization_coef 1e-07 --hidden_dim 200 --gamma 12.0 --lr 0.009 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 -de \
    --mix_cpu_gpu --num_thread 4 --max_step 40000 --gpu 0

# RotatE 8GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset FB15k --batch_size 2048 \
    --neg_sample_size 256 --regularization_coef 1e-07 --hidden_dim 200 --gamma 12.0 --lr 0.009 \
    --batch_size_eval 16 --valid --test -adv --mix_cpu_gpu --eval_interval 100000 -de \
    --mix_cpu_gpu --max_step 5000 --num_proc 8 --gpu 0 1 2 3 4 5 6 7 \
    --num_thread 4 --rel_part --async_update