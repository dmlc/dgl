# Graphbolt Quickstart Tutorial

Graphbolt provides all you need to create a dataloader to train a Graph Neural Networks.

## Dataset

The [cora/](https://github.com/dmlc/dgl/tree/master/examples/sampling/graphbolt/quickstart/cora)
dataset shows how to create a Graphbolt OnDiskDataset, where metadata.yaml is the catalog for all of
the data in the dataset.

## Examples

 - The [node_classification.py](https://github.com/dmlc/dgl/blob/master/examples/sampling/graphbolt/quickstart/node_classification.py)
   shows how to create a Graphbolt dataloader to train a 2 layer Graph Convolutional Networks node
   classification model.
 - The [link_prediction.py](https://github.com/dmlc/dgl/blob/master/examples/sampling/graphbolt/quickstart/link_prediction.py)
   shows how to create a Graphbolt dataloader to train a 2 layer GraphSage link prediction model.
