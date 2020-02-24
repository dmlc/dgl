MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 ../kvclient.py --model DistMult --dataset Freebase \
--batch_size 1000 --neg_sample_size 200 --hidden_dim 400 --gamma 143 --lr 0.08 --max_step 13000 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_test 1000 --test -adv --total_machine 4 --num_client 35