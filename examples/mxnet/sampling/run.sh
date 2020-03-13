export OMP_NUM_THREADS=4
export DGLBACKEND=mxnet
python3 examples/mxnet/sampling/dist_train.py --ip_config examples/mxnet/sampling/ip_config.txt --num-client 4 --id 0 --server_data reddit/server-0.dgl --server --model gcn_ns --n-classes 41 --num-neighbors 10 &
python3 examples/mxnet/sampling/dist_train.py --ip_config examples/mxnet/sampling/ip_config.txt --num-client 4 --id 1 --server_data reddit/server-1.dgl --server --model gcn_ns --n-classes 41 --num-neighbors 10 &
python3 examples/mxnet/sampling/dist_train.py --ip_config examples/mxnet/sampling/ip_config.txt --num-client 4 --id 2 --server_data reddit/server-2.dgl --server --model gcn_ns --n-classes 41 --num-neighbors 10 &
python3 examples/mxnet/sampling/dist_train.py --ip_config examples/mxnet/sampling/ip_config.txt --num-client 4 --id 3 --server_data reddit/server-3.dgl --server --model gcn_ns --n-classes 41 --num-neighbors 10 &

export OMP_NUM_THREADS=4
python3 /usr/local/lib/python3.6/dist-packages/mxnet/tools/launch.py -n 4 --launcher local python3 examples/mxnet/sampling/dist_train.py --ip_config examples/mxnet/sampling/ip_config.txt --graph-path reddit --model gcn_ns --n-classes 41 --num-neighbors 10 &
