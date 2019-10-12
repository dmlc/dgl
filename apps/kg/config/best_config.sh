#To reproduce reported results on README, you can run the model with the following commands:

# for FB15k

DGLBACKEND=pytorch python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 100000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv

DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.2 --max_step 100000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv

DGLBACKEND=pytorch python3 train.py --model TransE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --lr 0.01 --max_step 20000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv

DGLBACKEND=pytorch python3 train.py --model RESCAL --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 500 --gamma 24.0 --lr 0.03 --max_step 30000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv

# for wn18

DGLBACKEND=pytorch python3 train.py --model TransE --dataset wn18 --batch_size 1024 \
    --neg_sample_size 512 --hidden_dim 500 --gamma 12.0 --adversarial_temperature 0.5 \
    --lr 0.01 --max_step 40000 --batch_size_eval 16 --gpu 0 --valid --test -adv \
    --regularization_coef 0.00001

DGLBACKEND=pytorch python3 train.py --model DistMult --dataset wn18 --batch_size 1024 \
    --neg_sample_size 1024 --hidden_dim 1000 --gamma 200.0 --lr 0.1 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv --regularization_coef 0.00001

DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset wn18 --batch_size 1024 \
    --neg_sample_size 1024 --hidden_dim 500 --gamma 200.0 --lr 0.1 --max_step 20000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv --regularization_coef 0.00001

DGLBACKEND=pytorch python3 train.py --model RESCAL --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 250 --gamma 24.0 --lr 0.03 --max_step 20000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv

# for Freebase

DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 500.0     --lr 0.1 --max_step 50000 \
    --batch_size_eval 128 --test -adv --eval_interval 300000 \
    --neg_sample_size_test 100000 --eval_percent 0.02 --num_proc 64
