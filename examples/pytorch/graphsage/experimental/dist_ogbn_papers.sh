#!/usr/bin/bash

echo "Executing ogbn-papers dataset..."
#echo "Creating partitions, this might take a while as the code is sequential."
#python ../../../../python/dgl/distgnn/partition/main_Libra.py ogbn-papers100M

echo "Performing distributed runs:"
for i in 32 64 128
do
	echo "########################################################################################################"
    echo "Processing partition: "$i
    echo "cd-0 "$i
	sh run_dist.sh -n $i -ppn 1  python train_dist_sym_ogbn-papers.py --dataset ogbn-papers100M --n-epochs 200 --nr 1 --lr 0.08
    echo "cd-5 "$i
	sh run_dist.sh -n $i -ppn 1  python train_dist_sym_ogbn-papers.py --dataset ogbn-papers100M --n-epochs 200 --nr 5 --lr 0.08
    echo "0c "$i
	sh run_dist.sh -n $i -ppn 1  python train_dist_sym_ogbn-papers.py --dataset ogbn-papers100M --n-epochs 200 --nr -1 --lr 0.08
done
echo "Ogbn-papers experiment completed !!"

