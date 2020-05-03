export OMP_NUM_THREADS=4
export DGLBACKEND=mxnet
python3 train_dist.py --ip_config ~/dgl/ip_config.txt --num-client 2 --id 0 --server_data ~/dgl/reddit/server-0.dgl --server --client_data ~/dgl/reddit/client-0.dgl --graph-name reddit --model gcn_ns &
python3 train_dist.py --ip_config ~/dgl/ip_config.txt --num-client 2 --id 1 --server_data ~/dgl/reddit/server-1.dgl --server --client_data ~/dgl/reddit/client-1.dgl --graph-name reddit --model gcn_ns &

export OMP_NUM_THREADS=4
python3 train_dist.py --ip_config ~/dgl/ip_config.txt --model gcn_ns --n-classes 41 --n-features 602 --graph-name reddit &
