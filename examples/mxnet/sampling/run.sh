DGLBACKEND=mxnet python3 dist_train.py --id 0 --graph-name reddit --server 1 --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 &
DGLBACKEND=mxnet python3 dist_train.py --id 1 --graph-name reddit --server 1 --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 &
DGLBACKEND=mxnet python3 dist_train.py --id 2 --graph-name reddit --server 1 --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 &
DGLBACKEND=mxnet python3 dist_train.py --id 3 --graph-name reddit --server 1 --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 &

DGLBACKEND=mxnet python3 dist_train.py --id 0 --graph-name reddit --server 0 --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 &
DGLBACKEND=mxnet python3 dist_train.py --id 1 --graph-name reddit --server 0 --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 &
DGLBACKEND=mxnet python3 dist_train.py --id 2 --graph-name reddit --server 0 --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 &
DGLBACKEND=mxnet python3 dist_train.py --id 3 --graph-name reddit --server 0 --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 &
