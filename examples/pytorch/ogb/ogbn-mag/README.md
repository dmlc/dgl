## Running
The task can be run with default parameters as follows:  `python hetero_rgcn.py`

The following options can be specified via command line arguments:
```
optional arguments:
  -h, --help            show this help message and exit
  --runs RUNS
```

### Performance
Running the task with default parameters should yield performance similar to below:

```
Final performance:
All runs:
Highest Train: 84.67 ± 0.37
Highest Valid: 48.75 ± 0.39
  Final Train: 71.08 ± 7.09
   Final Test: 47.81 ± 0.37
```

This is a result of 10 experiments where each experiment is run for 3 epochs.  In the table above, "Highest" corresponds to the maximum value over the 3 epochs and "Final" corresponds to the value obtained when evaluating with the model parameters _as they were when the Validation accuracy was its maximum_.  For example, if the best Valid Accuracy was achieved at the end of epoch 2, then "Final Train" and "Final Test" are the Train and Test accuracies after epoch 2.  The values reported in the table are the average and standard deviations of these metrics from 10 runs.

Typically, the best Validation performance is obtained after the 1st or 2nd epoch, after which it begins to overfit.  This is why "Highest Train" (typically occuring at the end of the 3rd epoch), is significantly higher than "Final Train" (corresponding to epoch of maximal Validation performance).

## Background
The purpose of this example is to faithfully recreate the ogbn-mag NeighborSampling (R-GCN aggr) [PyG implementation](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/sampler.py) using DGL's HeteroGraph API.  This effort is a result of a deep-dive in [#3511](https://github.com/dmlc/dgl/issues/3511), which uncovered a number of differences between a simple R-GCN minibatch DGL implementation (e.g. like [this one](https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/entity_classify_mb.py)) and one specific to the OGB MAG dataset.

Some examples of such differences:
- Instead of reversing `(paper, cites, paper)` into a new relation like `(paper, rev-cites, paper)`, the PyG implementation instead just made these into undirected edges ([code](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/sampler.py#L54))
- In the PyG implementation there's a separate "self" linear projection matrix for each _node-type_ ([code](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/sampler.py#L106)).  This is different from the R-GCN [paper](https://arxiv.org/abs/1703.06103), which has a single "self" linear projection matrix for each R-GCN layer, not a different one for each node-type.

### Neighborhood sampling differences
Although the model architectures, hyperparameter values and initialization methods are identical between the implementation here and the PyG one as of this writing, there is still a significant difference in the way neighbors are sampled, which results in the DGL implementation achieving significantly faster overfitting to the training dataset and slightly improved performance on the Test dataset.

In DGL, sampling on heterogeneous graphs with a `fanout = N` parameter means there are N samples _per incoming relation type_.  In the PyG implementation, the heterogeneous graph is represented as a homogeneous graph and there are N samples total, regardless of relation type.  This effectively means that given the same `fanout` value, there are R times as many neighbors sampled for DGL than PyG, where R is the number of edge-types that are directed inward to a node.  Since there are significantly more nodes involved in the computation, there are likewise more nodes receiving gradient updates and therefore more significant overfitting given the same number of epochs.

An effort was made to mitigate this increase by reducing the fanout from `[25, 20]` to `[6, 5]`, which gives roughly the same number of neighbors between PyG and DGL and similar final training performance.  However, the DGL implementation has significantly worse Test performance in this case.  This is likely due to the fact that sampling e.g., 5 nodes from 4 different edge types is not the same as sampling 20 nodes by ignoring edge type unless the edge types are uniformly distributed.

### Input features
The `paper` nodes have 128-dimensional features that are derived from word embeddings of the words found in the title and abstract of the papers.  Following the PyG implementation, all node types except `paper` receive 128-dimensional learnable embeddings as node features.  This results in 154,029,312 learnable parameters for just the node features.

```
ParameterDict(
    (author): Parameter containing: [torch.FloatTensor of size 1134649x128]
    (field_of_study): Parameter containing: [torch.FloatTensor of size 59965x128]
    (institution): Parameter containing: [torch.FloatTensor of size 8740x128]
)
```

### Model architecture
The input features are passed to a modified version of the R-GCN architecture.  As in the R-GCN paper, each _edge-type_ has its own linear projection matrix (the "weight" ModuleDict below).  Different from the original paper, however, each _node-type_ has its own "self" linear projection matrix (the "loop_weights" ModuleDict below).  There are 7 edge-types:  4 natural edge-types ("cites", "affiliated_with", "has_topic" and "writes") and 3 manufactured reverse edge-types ("rev-affiliated_with", "rev-has_topic", "rev-writes").  As mentioned above, note that there is _not_ a reverse edge type like "rev-cites", and instead the reverse edges are given the same type of "cites".  This exception was presumably made because the source and destinate nodes are of type "paper".  Whereas the 7 "relation" linear layers do not have a bias, the 4 "self" linear layers do.

With two of these layers, a hidden dimension size of 64 and 349 output classes, we end up with 337,460 R-GCN model parameters.
