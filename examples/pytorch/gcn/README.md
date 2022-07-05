Graph Convolutional Networks (GCN)
============

- Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
- Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn).

How to run
-------

### DGL built-in GraphConv module

Run with the following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train.py --dataset cora
```

Summary
-------
* cora: ~0.810 (paper: 0.815)
* citeseer: ~0.707 (paper: 0.703)
* pubmed: ~0.792 (paper: 0.790)

