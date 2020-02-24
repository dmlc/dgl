COUNTER=0
while [ $COUNTER -lt 13 ]
do
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 ../kvserver.py --model DistMult --dataset Freebase \
    --hidden_dim 400 --gamma 143 --lr 0.08 --total_client 140 --server_id $COUNTER &
    let COUNTER+=1
done