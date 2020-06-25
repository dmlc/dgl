numactl -N 0 python3 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --num-client 8 &
numactl -N 1 python3 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --num-client 8 &
