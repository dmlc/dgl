#To reproduce reported results on README, you can run the model with the following commands:

DGLBACKEND=pytorch python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 100000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv

DGLBACKEND=pytorch python3 main.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.2 --max_step 100000 \
    --batch_size_eval 16 --gpu 1 --train --valid --test -adv

DGLBACKEND=pytorch python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --lr 0.01 --max_step 20000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv

