Predict then Propagate: Graph Neural Networks meet Personalized PageRank (APPNP)
============

- Paper link: [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997)
- Author's code repo: [https://github.com/klicperajo/ppnp](https://github.com/klicperajo/ppnp). 

Dependencies
------------
- MXNET 1.5+
- requests

``bash
pip install torch requests
``

Code
-----
The folder contains an implementation of APPNP (`appnp.py`).

Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
DGLBACKEND=mxnet python3 appnp.py --dataset cora --gpu 0
```

* cora: 0.8370 (paper: 0.850)
* citeseer: 0.713 (paper: 0.757)
* pubmed: 0.798 (paper: 0.797)

Experiments were done on dgl datasets (GCN settings) which are different from those used in the original implementation. (discrepancies are detailed in experimental section of the original paper)
