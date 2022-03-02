.. _api-dataloading:

dgl.dataloading
=================================

.. automodule:: dgl.dataloading
.. currentmodule:: dgl.dataloading

DataLoaders
-----------

DGL DataLoader for mini-batch training works similarly to PyTorch's DataLoader.
It has a generator interface that returns mini-batches sampled from some given graphs.
DGL provides two DataLoaders: a ``NodeDataLoader`` for node classification task
and an ``EdgeDataLoader`` for edge/link prediction task.

.. autosummary::
    :toctree: ../../generated/

    DataLoader
    NodeDataLoader
    EdgeDataLoader
    GraphDataLoader
    DistNodeDataLoader
    DistEdgeDataLoader

.. _api-dataloading-neighbor-sampling:

Samplers
--------

.. autosummary::
    :toctree: ../../generated/

    Sampler
    BlockSampler
    NeighborSampler
    MultiLayerFullNeighborSampler
    ClusterGCNSampler
    ShaDowKHopSampler

Sampler Transformations
-----------------------

.. autosummary::
    :toctree: ../../generated/

    as_edge_prediction_sampler

.. _api-dataloading-negative-sampling:

Negative Samplers for Link Prediction
-------------------------------------
.. currentmodule:: dgl.dataloading.negative_sampler

Negative samplers are classes that control the behavior of the edge prediction samplers

.. autosummary::
    :toctree: ../../generated/

    Uniform
    PerSourceUniform
    GlobalUniform

Utility Class and Functions for Feature Prefetching
---------------------------------------------------
.. currentmodule:: dgl.dataloading.base

.. autosummary::
    :toctree: ../../generated/

    LazyFeature
    set_node_lazy_features
    set_edge_lazy_features
    set_src_lazy_features
    set_dst_lazy_features
