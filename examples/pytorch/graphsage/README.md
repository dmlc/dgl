Inductive Representation Learning on Large Graphs (GraphSAGE)
============

- Paper link: [http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
- Author's code repo: [https://github.com/williamleif/graphsage-simple](https://github.com/williamleif/graphsage-simple). Note that the original code is 
simple reference implementation of GraphSAGE.

Advanced usages, including how to run pure GPU sampling, how to train with PyTorch Lightning, etc., are in the `advanced` directory.

Requirements
------------

```bash
pip install requests torchmetrics
```

Results
-------

### Full graph training

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train_full.py --dataset cora --gpu 0    # full graph
```

* cora: ~0.8330 
* citeseer: ~0.7110
* pubmed: ~0.7830

### Minibatch training for node classification

Train w/ mini-batch sampling for node classification on OGB-products:

```bash
python3 node_classification.py
python3 multi_gpu_node_classification.py
```

### PyTorch Lightning for node classification

Train w/ mini-batch sampling for node classification with PyTorch Lightning on OGB-products.
Works with both single GPU and multiple GPUs:

```bash
python3 lightning/node_classification.py
```

### Minibatch training for link prediction

Train w/ mini-batch sampling for link prediction on OGB-Citation2:

```bash
python3 link_pred.py
```
