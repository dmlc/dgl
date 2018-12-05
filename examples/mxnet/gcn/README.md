Graph Convolutional Networks (GCN)
============

Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn)

Codes
-----
The folder contains two implementations of GCN. `gcn_batch.py` uses user-defined
message and reduce functions. `gcn_spmv.py` uses DGL's builtin functions so
SPMV optimization could be applied.

Results
-------
These results are based on single-run training to minimize the cross-entropy
loss of the first 20 examples in each class. We can see clear improvements of
graph convolution networks (GCNs) over multi-layer perceptron (MLP) baselines.
There are also some slight modifications from the original paper:

* We used more (up to 10) layers to demonstrate monotonic improvements as more
  neighbor information is used. Using GCN with more layers improves accuracy
but can also increase the computational complexity. The original paper
recommends n-layers=2 to balance speed and accuracy.
* We used concatenation of hidden units to account for multi-hop
  skip-connections. The original implementation used simple additions (while
the original paper omitted this detail). We feel concatenation is superior
because all neighboring information is presented without additional modeling
assumptions.
* After the concatenation, we used a recursive model such that the (k+1)-th
  layer, storing information up to the (k+1)-distant neighbor, depends on the
concatenation of all 1-to-k layers. However, activation is only applied to the
new information in the concatenations.

```
# Final accuracy 75.34% MLP without GCN
DGLBACKEND=mxnet python examples/mxnet/gcn/gcn_batch.py --dataset "citeseer" --n-epochs 200 --gpu 1 --n-layers 0

# Final accuracy 86.57% with 10-layer GCN (symmetric normalization)
DGLBACKEND=mxnet python examples/mxnet/gcn/gcn_batch.py --dataset "citeseer" --n-epochs 200 --gpu 1 --n-layers 10 --normalization 'sym' --self-loop

# Final accuracy 84.42% with 10-layer GCN (unnormalized)
DGLBACKEND=mxnet python examples/mxnet/gcn/gcn_batch.py --dataset "citeseer" --n-epochs 200 --gpu 1 --n-layers 10
```

```
# Final accuracy 40.62% MLP without GCN
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_batch.py --dataset "cora" --n-epochs 200 --gpu 1 --n-layers 0

# Final accuracy 92.63% with 10-layer GCN (symmetric normalization)
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_batch.py --dataset "cora" --n-epochs 200 --gpu 1 --n-layers 10 --normalization 'sym' --self-loop

# Final accuracy 86.60% with 10-layer GCN (unnormalized)
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_batch.py --dataset "cora" --n-epochs 200 --gpu 1 --n-layers 10
```

```
# Final accuracy 72.97% MLP without GCN
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_batch.py --dataset "pubmed" --n-epochs 200 --gpu 1 --n-layers 0

# Final accuracy 88.33% with 10-layer GCN (symmetric normalization)
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_batch.py --dataset "pubmed" --n-epochs 200 --gpu 1 --n-layers 10 --normalization 'sym' --self-loop

# Final accuracy 83.80% with 10-layer GCN (unnormalized)
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_batch.py --dataset "pubmed" --n-epochs 200 --gpu 1 --n-layers 10
```
