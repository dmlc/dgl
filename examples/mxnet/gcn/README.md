Graph Convolutional Networks (GCN)
============

Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn)

Dependencies
------------
- MXNet nightly build
- requests

``bash
pip install mxnet --pre
pip install requests
``

Codes
-----
The folder contains three implementations of GCN:
- `gcn.py` uses DGL's predefined graph convolution module.
- `gcn_mp.py` uses user-defined message and reduce functions.
Modify `train.py` to switch between different implementations.

The provided implementation in `gcn_concat.py` is a bit different from the
original paper for better performance, credit to @yifeim and @ZiyueHuang.

Results
-------
Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
DGLBACKEND=mxnet python3 train.py --dataset cora --gpu 0 --self-loop
```

* cora: ~0.810 (paper: 0.815)
* citeseer: ~0.702 (paper: 0.703)
* pubmed: ~0.780 (paper: 0.790)

Results (`gcn_concat.py vs. gcn.py`)
------------------------------------
`gcn_concat.py` uses concatenation of hidden units to account for multi-hop
  skip-connections. We feel concatenation is superior
because all neighboring information is presented without additional modeling
assumptions.
These results are based on single-run training to minimize the cross-entropy
loss. We can see clear skip connection can help train a GCN with many layers.

The experiments show that adding depth may or may not improve accuracy.
While adding depth is a clear way to mimic power iterations of matrix factorizations,
training multiple epochs to obtain stationary points could equivalently solve matrix
factorization. Given the small datasets, we can't draw such conclusions from these experiments.

```
# Final accuracy 57.70% MLP without GCN
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_concat.py --dataset "citeseer" --n-epochs 200 --n-layers 0

# Final accuracy 65.70% with 10-layer GCN with skip connection
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_concat.py --dataset "citeseer" --n-epochs 200 --n-layers 2 --normalization 'sym' --self-loop

# Final accuracy 64.70% with 10-layer GCN with skip connection
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_concat.py --dataset "citeseer" --n-epochs 200 --n-layers 10 --normalization 'sym' --self-loop

```

```
# Final accuracy 53.20% MLP without GCN
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_concat.py --dataset "cora" --n-epochs 200 --n-layers 0

# Final accuracy 72.60% with 2-layer GCN with skip connection
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_concat.py --dataset "cora" --n-epochs 200 --n-layers 2 --normalization 'sym' --self-loop

# Final accuracy 78.90% with 10-layer GCN with skip connection
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_concat.py --dataset "cora" --n-epochs 200 --n-layers 10 --normalization 'sym' --self-loop

```

```
# Final accuracy 70.30% MLP without GCN
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_concat.py --dataset "pubmed" --n-epochs 200 --n-layers 0

# Final accuracy 78.30% with 2-layer GCN with skip connection
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_concat.py --dataset "pubmed" --n-epochs 200 --n-layers 2 --normalization 'sym' --self-loop

# Final accuracy 76.30% with 10-layer GCN with skip connection
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_concat.py --dataset "pubmed" --n-epochs 200 --n-layers 10 --normalization 'sym' --self-loop
```
