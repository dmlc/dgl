export OMP_NUM_THREADS=4
export DGLBACKEND=mxnet
python3 dist_train.py --id 0 --graph-name reddit --server --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 --num-neighbors 10 &
python3 dist_train.py --id 1 --graph-name reddit --server --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 --num-neighbors 10 &
python3 dist_train.py --id 2 --graph-name reddit --server --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 --num-neighbors 10 &
python3 dist_train.py --id 3 --graph-name reddit --server --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 --num-neighbors 10 &

export OMP_NUM_THREADS=4
~/mxnet/tools/launch.py -n 4 --launcher local python3 dist_train.py --id 0 --graph-name reddit --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 --num-neighbors 10 &
#python3 dist_train.py --id 0 --graph-name reddit --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 --num-neighbors 10 &
#python3 dist_train.py --id 1 --graph-name reddit --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 --num-neighbors 10 &
#python3 dist_train.py --id 2 --graph-name reddit --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 --num-neighbors 10 &
#python3 dist_train.py --id 3 --graph-name reddit --num-parts 4 --model gcn_ns --n-classes 41 --n-features 602 --num-neighbors 10 &

#DGLBACKEND=mxnet python3 dist_train.py --id 0 --graph-name reddit --server 1 --num-parts 1 --model gcn_ns --n-classes 41 --n-features 602 --dataset reddit &
#DGLBACKEND=mxnet python3 dist_train.py --id 0 --graph-name reddit --server 0 --num-parts 1 --model gcn_ns --n-classes 41 --n-features 602 --dataset reddit &
