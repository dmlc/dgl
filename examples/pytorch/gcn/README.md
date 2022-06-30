Graph Convolutional Networks (GCN)
============

- Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
- Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn). Note that the original implementation is based on Tensorflow.

Requirements
------------
```
pip install torch requests
```

Available examples
-----
The folder contains two implementations of GCN:
- DGL intrinsic graph convolution module ([train.py](https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/train.py))
- Customized GCN layer with user-defined message and reduce functions ([gcn_mp.py](https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/gcn_mp.py))

How to run
-------

### DGL intrinsic GCN module

Run with the following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train.py --dataset cora --self-loop
```
Results:
```
Test Accuracy: 0.8140
```

### GCN with User-defined message and reduce functions

Run with the following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 gcn_mp.py --dataset cora --self-loop
```
Results:
```
Test Accuracy: 0.8150
```

Summary
-------
* cora: ~0.810 (0.79-0.83) (paper: 0.815)
* citeseer: 0.707 (paper: 0.703)
* pubmed: 0.792 (paper: 0.790)

