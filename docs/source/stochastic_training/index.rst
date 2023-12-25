🆕 Stochastic Training of GNNs with GraphBolt
=============================================

GraphBolt is a data loading framework for GNN with high flexibility and
scalability. It is built on top of DGL and PyTorch.

This tutorial introduces how to enable stochastic training of GNNs with
GraphBolt.

Overview
^^^^^^^

.. image:: ../../_static/graphbolt_overview.png
  :width: 700
  :alt: Graphbolt Overview

GraphBolt seamlessly integrates with the PyTorch `datapiple <https://pytorch.org/data/beta/torchdata.datapipes.iter.html>`_, where each stage's output is just the input of its successor, streamlining data loading and preprocessing for GNN training, validation and testing.
By default, GraphBolt provides a collection of built-in datasets and exceptionally efficient implementations of datapipes for common scenarios, which can be summarized as follows:

1. **Mini-Batch Sampler:** Randomly selects a subset (nodes, edges, graphs) from the entire training set to calculate embeddings.

2. **Negative Sampler:** In link prediction tasks, samples non-existing edges as negative examples to train the GNN in distinguishing between positive and negative edges.

3. **Subgraph Sampler:** Utilizes the output from mini-batch sampling to generate neighbor nodes up to a specified depth or distance.

4. **Feature Fetcher:** Fetch related node/edge features from the dataset.

By exposing the entire data loading process as a pipeline, GraphBolt provides significant flexibility and customization opportunities. Users can easily substitute any stage with their own implementations. Additionally, users can benefit from the optimized scheduling strategy for datapipes, even with customized stages.

In summary, GraphBolt offers the following benefits:

1. A flexible, pipelined framework for GNN data loading and preprocessing.

2. Highly efficient canonical implementations.

3. Efficient scheduling.

Scenarios
^^^^^^^

.. toctree::
  :maxdepth: 1

  neighbor_sampling_overview.nblink
  node_classification.nblink
  link_prediction.nblink
  ondisk-dataset.rst
