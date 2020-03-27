export OMP_NUM_THREADS=4
export DGLBACKEND=mxnet
python3 examples/mxnet/sampling/dist_train.py --ip_config ../ip_config.txt --num-client 2 --id 0 --server_data ../reddit/server-0.dgl --server --client_data ../reddit/client-0.dgl --graph-name reddit --model gcn_ns &
python3 examples/mxnet/sampling/dist_train.py --ip_config ../ip_config.txt --num-client 2 --id 1 --server_data ../reddit/server-1.dgl --server --client_data ../reddit/client-1.dgl --graph-name reddit --model gcn_ns &

export OMP_NUM_THREADS=4
python3 /usr/local/lib/python3.6/dist-packages/mxnet/tools/launch.py -n 2 --launcher ssh -H hosts.txt python3 examples/mxnet/sampling/dist_train.py --ip_config ../ip_config.txt --model gcn_ns --n-classes 41 --num-neighbors 10 --n-features 602 --graph-name reddit &
