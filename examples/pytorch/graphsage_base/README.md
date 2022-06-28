Inductive Representation Learning on Large Graphs (GraphSAGE)
============

- Paper link: [http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
- Author's code repo: [https://github.com/williamleif/graphsage-simple](https://github.com/williamleif/graphsage-simple). Note that the original code is 
simple reference implementation of GraphSAGE.

Advanced usages, including how to run pure GPU sampling, how to train with PyTorch Lightning, etc., are in the `graphsage/advanced` directory.

Requirements
------------

```bash
pip install requests torchmetrics
```

How to run
-------

### Mini-batch sampling for node classification
Train w/ mini-batch sampling for node classification on OGB-products:

```bash
# mini-batch training on gpu
python3 node_classification.py --gpu 0
```

Results:
```
Test Accuracy: 0.7632
```
