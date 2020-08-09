Inductive Representation Learning on Large Graphs (GraphSAGE)
============

- Paper link: [http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
- Author's code repo: [https://github.com/williamleif/graphsage-simple](https://github.com/williamleif/graphsage-simple). Note that the original code is 
simple reference implementation of GraphSAGE.

Requirements
------------
- requests

``bash
pip install requests
``


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

### Minibatch training

Train w/ mini-batch sampling (on the Reddit dataset)
```bash
python3 train_sampling.py --num-epochs 30       # neighbor sampling
python3 train_sampling.py --num-epochs 30 --inductive  # inductive learning with neighbor sampling
python3 train_sampling_multi_gpu.py --num-epochs 30    # neighbor sampling with multi GPU
python3 train_sampling_multi_gpu.py --num-epochs 30 --inductive  # inductive learning with neighbor sampling, multi GPU
python3 train_cv.py --num-epochs 30             # control variate sampling
python3 train_cv_multi_gpu.py --num-epochs 30   # control variate sampling with multi GPU
```

Accuracy:

| Model                 | Accuracy |
|:---------------------:|:--------:|
| Full Graph            | 0.9504   |
| Neighbor Sampling     | 0.9495   |
| N.S. (Inductive)      | 0.9460   |
| Control Variate       | 0.9490   |

### Unsupervised training

Train w/ mini-batch sampling in an unsupervised fashion (on the Reddit dataset)
```bash
python3 train_sampling_unsupervised.py
```

Notably,

* The loss function is defined by predicting whether an edge exists between two nodes or not.  This matches the official
  implementation, and is equivalent to the loss defined in the paper with 1-hop random walks.
* When computing the score of `(u, v)`, the connections between node `u` and `v` are removed from neighbor sampling.
  This trick increases the F1-micro score on test set by 0.02.
* The performance of the learned embeddings are measured by training a softmax regression with scikit-learn, as described
  in the paper.

Micro F1 score reaches 0.9212 on test set.
