#!/usr/bin/bash

echo "Executing ogbn-products dataset..."
#echo "Creating partitions, this might take a while as the code is sequential."
#python ../../../../python/dgl/distgnn/partition/main_Libra.py ogbn-products

echo "Performing distributed runs:"
for i in 2 4 8 16 32 64
do
	echo "########################################################################################################"
    echo "Processing partition: "$i
    echo "cd-0 "$i
    sh run_dist.sh -n $i -ppn 1  python train_dist_sym_ogbn-products.py --dataset ogbn-products --n-epochs 300 --nr 1 --lr 0.03
    echo "cd-5 "$i
    sh run_dist.sh -n $i -ppn 1  python train_dist_sym_ogbn-products.py --dataset ogbn-products --n-epochs 300 --nr 5 --lr 0.03
    echo "0c "$i
	sh run_dist.sh -n $i -ppn 1  python train_dist_sym_ogbn-products.py --dataset ogbn-products --n-epochs 300 --nr -1 --lr 0.03

done
echo "Ogbn-products experiment completed !!"

