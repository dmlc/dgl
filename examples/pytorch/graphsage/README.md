Inductive Representation Learning on Large Graphs (GraphSAGE)
============

- Paper link: [http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
- Author's code repo: [https://github.com/williamleif/graphsage-simple](https://github.com/williamleif/graphsage-simple)

For advanced usages, including training with multi-gpu/multi-node, and PyTorch Lightning, etc., more examples can be found in [advanced](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage/advanced) and [dist](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage/dist) directory.

Requirements
------------

```bash
pip install requests torchmetrics==0.11.4 ogb
```

How to run
-------

### Full graph training

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train_full.py --dataset cora --gpu 0    # full graph
```

Results:
```
* cora: ~0.8330
* citeseer: ~0.7110
* pubmed: ~0.7830
```

### Minibatch training for node classification

Train w/ mini-batch sampling in mixed mode (CPU+GPU) for node classification on "ogbn-products"

```bash
python3 node_classification.py
```

Results:
```
Test Accuracy: 0.7632
```

### PyTorch Lightning for node classification

Train w/ mini-batch sampling for node classification with PyTorch Lightning on OGB-products. It requires PyTorch Lightning 2.0.1. It works with both single GPU and multiple GPUs:

```bash
python3 lightning/node_classification.py
```

### Minibatch training for link prediction

Train w/ mini-batch sampling for link prediction on OGB-citation2:

```bash
python3 link_pred.py
```

Results (10 epochs):
```
Test MRR: 0.7386
```
