Graph Convolutional Networks (GCN)
============

- Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
- Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn). Note that the original implementation is based on Tensorflow.

Available examples
-----
The folder includes an implementations of GCN using DGL intrinsic graph convolution module.

How to run
-------

### DGL intrinsic GCN module

Run with the following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train.py --dataset cora
```
Results:
```
Test Accuracy: 0.8140
```

Summary
-------
* cora: ~0.810 (0.79-0.83) (paper: 0.815)
* citeseer: 0.707 (paper: 0.703)
* pubmed: 0.792 (paper: 0.790)

