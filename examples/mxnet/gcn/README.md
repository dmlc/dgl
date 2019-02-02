Graph Convolutional Networks (GCN)
============

Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn)

Requirements
------------
- requests

``bash
pip install requests
``

Codes
-----
The folder contains two implementations of GCN. `gcn.py` uses user-defined
message and reduce functions. `gcn_spmv.py` uses DGL's builtin functions so
SPMV optimization could be applied.

The provided implementation in `gcn_concat.py` is a bit different from the
original paper, credit to @yifeim and @ZiyueHuang. This model uses concatenation
of hidden units to account for multi-hop skip-connections, which helps
the models with many layers.

Results
-------
Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
DGLBACKEND=mxnet python gcn_spmv.py --dataset cora --gpu 0
```

* cora: ~0.810 (paper: 0.815)
* citeseer: ~0.702 (paper: 0.703)
* pubmed: ~0.780 (paper: 0.790)
