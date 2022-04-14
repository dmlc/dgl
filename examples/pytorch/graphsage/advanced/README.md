More Examples for Training GraphSAGE
============================

### Pure GPU sampling

```bash
python3 pure_gpu_node_classification.py
```

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

This example also demonstrates the advanced usages of multi-GPU `DDP` training, UVA-base sampling, full GPU sampling, and fine control of storing the graph structure and features individually.

### Training with PyTorch Lightning

We also provide minibatch training scripts with PyTorch Lightning in `train_lightning.py` and `train_lightning_unsupervised.py`.

Requires `pytorch_lightning` and `torchmetrics`.

```bash
python3 train_lightning.py
python3 train_lightning_unsupervised.py
```
