DGLBACKEND=mxnet python3 gcn_ns_sampler.py --ip 127.0.0.1:2049 --num-sender=1 --dataset reddit-self-loop --num-neighbors 3 --batch-size 1000 --test-batch-size 5000
