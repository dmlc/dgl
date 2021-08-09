#!/usr/bin/bash

echo "Processing reddit dataset"
echo "Creating partitions, this might take a while as the code is sequential."
python ../../../../python/dgl/distgnn/partition/main_Libra.py reddit

echo "Performing distributed runs:"
for i in 2 4 8 16
do
	echo "########################################################################################################"	
    echo "Processing partition: "$i
    echo "cd-0 "$i
    sh run_dist.sh -n $i -ppn 1  python train_dist_sym.py --dataset reddit --n-epochs 200 --nr 1  --lr 0.03
    echo "cd-5 "$i
    sh run_dist.sh -n $i -ppn 1  python train_dist_sym.py --dataset reddit --n-epochs 200 --nr 5  --lr 0.03
    echo "0c "$i
    sh run_dist.sh -n $i -ppn 1  python train_dist_sym.py --dataset reddit --n-epochs 200 --nr -1  --lr 0.03

done
echo "Reddit experiment completed !!"
