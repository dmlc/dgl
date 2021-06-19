#To reproduce reported results on README, you can run the model with the following commands:

# for FB15k
# DistMult 1GPU
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 \
    --valid --test -adv --gpu 0 --max_step 40000

# DistMult 8GPU
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 \
    --valid --test -adv --max_step 5000 --mix_cpu_gpu --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --async_update --soft_rel_part --force_sync_interval 1000

# ComplEx 1GPU
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 1024 --hidden_dim 400 --gamma 143.0 --lr 0.1 \
    --regularization_coef 2.00E-06 --batch_size_eval 16 --valid --test -adv --gpu 0 \
    --max_step 32000

# ComplEx 8GPU
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 1024 --hidden_dim 400 --gamma 143.0 --lr 0.1 \
    --regularization_coef 2.00E-06 --batch_size_eval 16 --valid --test -adv \
    --max_step 4000 --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --soft_rel_part --force_sync_interval 1000

# TransE_l1 1GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l1 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 64 --regularization_coef 1e-07 --hidden_dim 400 --gamma 16.0 \
    --lr 0.01 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 48000

# TransE_l1 8GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l1 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 64 --regularization_coef 1e-07 --hidden_dim 400 --gamma 16.0 \
    --lr 0.01 --batch_size_eval 16 --valid --test -adv --max_step 6000 --mix_cpu_gpu \
    --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --soft_rel_part \
    --force_sync_interval 1000

# TransE_l2 1GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef=1e-9 --hidden_dim 400 --gamma 19.9 \
    --lr 0.25 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 32000

# TransE_l2 8GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef=1e-9 --hidden_dim 400 --gamma 19.9 \
    --lr 0.25 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 4000 \
    --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --soft_rel_part \
    --force_sync_interval 1000

# RESCAL 1GPU
DGLBACKEND=pytorch python3 train.py --model RESCAL --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 500 --gamma 24.0 --lr 0.03 --batch_size_eval 16 \
    --gpu 0 --valid --test -adv --max_step 30000

# RESCAL 8GPU
DGLBACKEND=pytorch python3 train.py --model RESCAL --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 500 --gamma 24.0 --lr 0.03 --batch_size_eval 16 \
    --valid --test -adv --max_step 4000 --mix_cpu_gpu --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --async_update --soft_rel_part --force_sync_interval 1000

# TransR 1GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 5e-8 --hidden_dim 200 --gamma 8.0 \
    --lr 0.015 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 32000

# TransR 8GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 5e-8 --hidden_dim 200 --gamma 8.0 \
    --lr 0.015 --batch_size_eval 16 --valid --test -adv --max_step 4000 --mix_cpu_gpu \
    --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --soft_rel_part \
    --force_sync_interval 1000

# RotatE 1GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset FB15k --batch_size 2048 \
    --neg_sample_size 256 --regularization_coef 1e-07 --hidden_dim 200 --gamma 12.0 \
    --lr 0.009 --batch_size_eval 16 --valid --test -adv -de --max_step 20000 \
    --neg_deg_sample --gpu 0

# RotatE 8GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 1e-07 --hidden_dim 200 --gamma 12.0 \
    --lr 0.009 --batch_size_eval 16 --valid --test -adv -de --max_step 2500 \
    --neg_deg_sample --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --soft_rel_part --force_sync_interval 1000

# for wn18
# DistMult 1GPU 
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset wn18 --batch_size 2048 \
    --neg_sample_size 128 --regularization_coef 1e-06 --hidden_dim 512 --gamma 20.0 \
    --lr 0.14 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 20000

# DistMult 8GPU 
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset wn18 --batch_size 2048 \
    --neg_sample_size 128 --regularization_coef 1e-06 --hidden_dim 512 --gamma 20.0 \
    --lr 0.14 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 2500 \
    --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --force_sync_interval 1000

# ComplEx 1GPU
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset wn18 --batch_size 1024 \
    --neg_sample_size 1024 --regularization_coef 0.00001 --hidden_dim 512 --gamma 200.0 \
    --lr 0.1 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 20000

# ComplEx 8GPU 
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset wn18 --batch_size 1024 \
    --neg_sample_size 1024 --regularization_coef 0.00001 --hidden_dim 512 --gamma 200.0 \
    --lr 0.1 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 2500 \
    --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --force_sync_interval 1000

# TransE_l1 1GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l1 --dataset wn18 --batch_size 2048 \
    --neg_sample_size 128 --regularization_coef 2e-07 --hidden_dim 512 --gamma 12.0 \
    --lr 0.007 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 32000

# TransE_l1 8GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l1 --dataset wn18 --batch_size 2048 \
    --neg_sample_size 128 --regularization_coef 2e-07 --hidden_dim 512 --gamma 12.0 \
    --lr 0.007 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 4000 \
    --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --force_sync_interval 1000

# TransE_l2 1GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 0.0000001 --hidden_dim 512 --gamma 6.0 \
    --lr 0.1 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 32000

# TransE_l2 8GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 0.0000001 --hidden_dim 512 --gamma 6.0 \
    --lr 0.1 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 4000 \
    --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --force_sync_interval 1000

# RESCAL 1GPU
DGLBACKEND=pytorch python3 train.py --model RESCAL --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 250 --gamma 24.0 --lr 0.03 --batch_size_eval 16 \
    --valid --test -adv --gpu 0 --max_step 20000

# RESCAL 8GPU
DGLBACKEND=pytorch python3 train.py --model RESCAL --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 250 --gamma 24.0 --lr 0.03 --batch_size_eval 16 \
    --valid --test -adv --gpu 0 --max_step 2500  --mix_cpu_gpu --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --async_update --force_sync_interval 1000 --soft_rel_part

# TransR 1GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 250 --gamma 16.0 --lr 0.1 --batch_size_eval 16 \
    --valid --test -adv --gpu 0 --max_step 30000

# TransR 8GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 250 --gamma 16.0 --lr 0.1 --batch_size_eval 16 \
    --valid --test -adv --max_step 2500  --mix_cpu_gpu --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --async_update --force_sync_interval 1000 --soft_rel_part

# RotatE 1GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset wn18 --batch_size 2048 \
    --neg_sample_size 64 --regularization_coef 2e-07 --hidden_dim 256 --gamma 9.0 \
    --lr 0.0025 -de --batch_size_eval 16 --neg_deg_sample --valid --test -adv --gpu 0 \
    --max_step 24000 

# RotatE 8GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset wn18 --batch_size 2048 \
    --neg_sample_size 64 --regularization_coef 2e-07 --hidden_dim 256 --gamma 9.0 \
    --lr 0.0025 -de --batch_size_eval 16 --neg_deg_sample --valid --test -adv \
    --max_step 3000 --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --force_sync_interval 1000

# for Freebase multi-process-cpu
# TransE_l2
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset Freebase --batch_size 1000 \
    --neg_sample_size 200 --hidden_dim 400 --gamma 10 --lr 0.1 --max_step 50000 \
    --log_interval 100 --batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv \
    --regularization_coef 1e-9 --num_thread 1 --num_proc 48

# DistMult
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --max_step 50000 \
    --log_interval 100 --batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv \
    --num_thread 1 --num_proc 48

# ComplEx
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.1 --max_step 50000 \
    --log_interval 100 --batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv \
    --num_thread 1 --num_proc 48

# Freebase multi-gpu
# TransE_l2 8GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset Freebase --batch_size 1000 \
    --neg_sample_size 200 --hidden_dim 400 --gamma 10 --lr 0.1 --regularization_coef 1e-9 \
    --batch_size_eval 1000 --valid --test -adv --mix_cpu_gpu --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --max_step 320000 --neg_sample_size_eval 1000 --eval_interval \
    100000 --log_interval 10000 --async_update --soft_rel_part --force_sync_interval 10000

# DistMult 8GPU
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 1000 \
    --valid --test -adv --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --max_step 300000 \
    --neg_sample_size_eval 1000 --eval_interval 100000 --log_interval 10000 --async_update \
    --soft_rel_part --force_sync_interval 10000

# ComplEx 8GPU
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143 --lr 0.1 \
    --regularization_coef 2.00E-06 --batch_size_eval 1000 --valid --test -adv \
    --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --max_step 360000 \
    --neg_sample_size_eval 1000 --eval_interval 100000 --log_interval 10000 \
    --async_update --soft_rel_part --force_sync_interval 10000

# TransR 8GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 5e-8 --hidden_dim 200 --gamma 8.0 \
    --lr 0.015 --batch_size_eval 1000 --valid --test -adv --mix_cpu_gpu --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --max_step 300000 --neg_sample_size_eval 1000 \
    --eval_interval 100000 --log_interval 10000 --async_update --soft_rel_part \
    --force_sync_interval 10000

# RotatE 8GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 -de --hidden_dim 200 --gamma 12.0 --lr 0.01 \
    --regularization_coef 1e-7 --batch_size_eval 1000 --valid --test -adv --mix_cpu_gpu \
    --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --max_step 300000 --neg_sample_size_eval 1000 \
    --eval_interval 100000 --log_interval 10000 --async_update --soft_rel_part \
    --force_sync_interval 10000

