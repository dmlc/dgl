Inductive Representation Learning on Large Graphs (GraphSAGE)
============

- Paper link: [http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
- Author's code repo: [https://github.com/williamleif/graphsage-simple](https://github.com/williamleif/graphsage-simple). Note that the original code is 
simple reference implementation of GraphSAGE.

Requirements
------------
- requests

```bash
pip install requests
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

### Minibatch training

Train w/ mini-batch sampling (on the Reddit dataset)
```bash
python3 train_sampling.py --num-epochs 30       # neighbor sampling
python3 train_sampling.py --num-epochs 30 --inductive  # inductive learning with neighbor sampling
python3 train_cv.py --num-epochs 30             # control variate sampling
```

For multi-gpu training
```bash
python3 train_sampling_multi_gpu.py --num-epochs 30 --gpu 0,1,...    # neighbor sampling
python3 train_sampling_multi_gpu.py --num-epochs 30 --inductive --gpu 0,1,...  # inductive learning
python3 train_cv_multi_gpu.py --num-epochs 30 --gpu 0,1,...   # control variate sampling
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

### Use GPU sampling and CUDA UVA sampling

For training scripts `train_sampling.py`, `train_sampling_multi_gpu.py` and `train_sampling_unsupervised.py`, we provide arguments `--graph-device` and `--data-device`.

For `--graph-device`, we provide the following choices:
- `cpu` (default): Use CPU to sample the graph structure stored in host memory.
- `gpu`: Use GPU to sample the graph structure stored in GPU device memory. You have to copy the graph structure (only the `csc` format is needed) to GPU before passing it to the dataloader. This is the fastest way for sampling but requires storing the whole graph structure in GPU memory and will duplicate it in each GPU in multi-GPU training.
- `uva`: Use GPU to sample the graph structure stored in **pinned** host memory through zero-copy access. You have to pin the graph structure before passing it to the dataloader. This is much faster than CPU sampling and especially useful when the graph structure is too large to fit into the GPU memory.

For `--data-device`, we provide the following choices:
- `cpu`: Node features are stored in host memory. It will take a lot time for slicing and transfering node features to GPU during training.
- `gpu` (default): Node features are stored in GPU device memory. This is the fastest way for feature slicing and transfering but cosumes a lot of GPU memory.
- `uva`: Use GPU to slice and access the node features stored in **pinned** host memory (also called `UnifiedTensor`) through zero-copy access. This is especially useful when the node features are too large to fit into the GPU memory.

### Training with PyTorch Lightning

We also provide minibatch training scripts with PyTorch Lightning in `train_lightning.py` and `train_lightning_unsupervised.py`.

Requires `pytorch_lightning` and `torchmetrics`.

```bash
python3 train_lightning.py
python3 train_lightning_unsupervised.py
```
