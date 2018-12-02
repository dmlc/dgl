# Benchmark SSE on multi-GPUs
# Use a small embedding.
DGLBACKEND=mxnet python3 examples/mxnet/sse/sse_batch.py --graph-file ../../data/5_5_csr.nd  --n-epochs 1 --lr 0.0005 --batch-size 1024 --use-spmv --dgl --num-parallel-subgraphs 32 --gpu 1 --num-feats 100 --n-hidden 100

# test convergence
DGLBACKEND=mxnet python3 examples/mxnet/sse/sse_batch.py --dataset "pubmed" --n-epochs 100 --lr 0.001 --batch-size 1024 --dgl --use-spmv --neigh-expand 4
