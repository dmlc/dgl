MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 ../worker.py --model TransE_l2 --dataset Freebase \
--batch_size 1000 --neg_sample_size 200 --hidden_dim 400 --gamma 10 --lr 0.1 --max_step 200 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_test 1000 --test -adv --regularization_coef 1e-9 \
--total_machine 4 --num_client 35