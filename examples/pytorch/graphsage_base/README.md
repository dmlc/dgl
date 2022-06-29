Inductive Representation Learning on Large Graphs (GraphSAGE)
============

- Paper link: [http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
- Author's code repo: [https://github.com/williamleif/graphsage-simple](https://github.com/williamleif/graphsage-simple)

For advanced usages, including training with multi-gpu/multi-node, and PyTorch Lightning, etc., more examples can be found in [GraphSAGE advanced](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage/advanced) directory.

Requirements
------------

```bash
pip install requests torchmetrics
```

How to run
-------

### Mini-batch sampling for node classification
Train w/ mini-batch sampling for node classification on "ogbn-products" 

```bash
# mini-batch training with mixed (CPU+GPU) mode
python3 node_classification.py
```

Results:
```
Test Accuracy: 0.7632
```
