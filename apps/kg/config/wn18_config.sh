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