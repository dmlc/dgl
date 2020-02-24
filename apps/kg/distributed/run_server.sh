COUNTER=0
while [ $COUNTER -lt 13 ]
do
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 ../kvserver.py --model TransE_l2 --dataset Freebase \
    --hidden_dim 400 --gamma 10 --lr 0.1 --num_client 140 --server_id $COUNTER &
    let COUNTER+=1
done