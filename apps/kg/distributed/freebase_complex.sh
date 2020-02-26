SERVER_ID_LOW=$1
SERVER_ID_HIGH=$2

while [ $SERVER_ID_LOW -lt $SERVER_ID_HIGH ]
do
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 ../kvserver.py --model ComplEx --dataset Freebase \
    --hidden_dim 400 --gamma 500.0 --lr 0.1 --total_client 140 --server_id $SERVER_ID_LOW &
    let SERVER_ID_LOW+=1
done

MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 ../kvclient.py --model ComplEx --dataset Freebase \
--batch_size 1000 --neg_sample_size 200 --hidden_dim 400 --gamma 500.0 --lr 0.1 --max_step 12500 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_test 1000 --test -adv --total_machine 4 --num_client 35 &