Benchmark SSE on multi-GPUs
=======================

Paper link:
[http://proceedings.mlr.press/v80/dai18a/dai18a.pdf](http://proceedings.mlr.press/v80/dai18a/dai18a.pdf)

Dependencies
-------------
* MXNet nightly build
* requests

```bash
pip install mxnet --pre
pip install requests
```

Use a small embedding
---------------------

```bash
DGLBACKEND=mxnet python3 sse_batch.py --graph-file ../../data/5_5_csr.nd \
                                      --n-epochs 1 \
                                      --lr 0.0005 \
                                      --batch-size 1024 \
                                      --use-spmv \
                                      --num-parallel-subgraphs 32 \
                                      --gpu 1 \
                                      --num-feats 100 \
                                      --n-hidden 100
```

Test convergence
----------------

```bash
DGLBACKEND=mxnet python3 sse_batch.py --dataset "pubmed" \
                                      --n-epochs 1000 \
                                      --lr 0.001 \
                                      --batch-size 30 \
                                      --use-spmv \
                                      --neigh-expand 8 \
                                      --n-hidden 16
```
