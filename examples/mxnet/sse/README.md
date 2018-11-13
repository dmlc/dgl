# Benchmark SSE on multi-GPUs
# Use a small embedding.
DGLBACKEND=mxnet python3 -m pyinstrument -o prof.out examples/mxnet/sse/sse_batch.py --graph-file ../../data/5_5_csr.nd  --n-epochs 1 --lr 0.0005 --batch-size 1024 --use-spmv 1 --num-parallel-subgraphs 32 --gpu 8
# Use a large embedding.
DGLBACKEND=mxnet python3 examples/mxnet/sse/sse_batch.py --graph-file ../../data/5_5_csr.nd  --n-epochs 1 --lr 0.0005 --batch-size 2048 --use-spmv 1 --num-parallel-subgraphs 32 --gpu 8 --n-hidden 500
