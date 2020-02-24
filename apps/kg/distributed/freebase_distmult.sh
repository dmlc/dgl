SERVER_ID_LOW=$1
SERVER_ID_HIGH=$2

while [ $SERVER_ID_LOW -lt $SERVER_ID_HIGH ]
do
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 ../kvserver.py --model DistMult --dataset Freebase \
    --hidden_dim 400 --gamma 143 --lr 0.08 --total_client 140 --server_id $SERVER_ID_LOW &
    let SERVER_ID_LOW+=1
done

MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 ../kvclient.py --model DistMult --dataset Freebase \
--batch_size 1000 --neg_sample_size 200 --hidden_dim 400 --gamma 143 --lr 0.08 --max_step 200 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_test 1000 --test -adv --total_machine 4 --num_client 35 &