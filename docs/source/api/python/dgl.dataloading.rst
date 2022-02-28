.. _api-dataloading:

dgl.dataloading
=================================

.. currentmodule:: dgl.dataloading

The ``dgl.dataloading`` package contains:

* ``DataLoader`` classes for iterating over a set of nodes or edges in a graph and generates
  sampled mini-batches.

* Various ``Sampler`` classes that extract subgraphs and sub-features.

* Negative samplers for link prediction.

The user guide chapter :ref:`guide-minibatch` explains how different components work together.

DataLoaders
-----------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

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
    :nosignatures:
    :template: classtemplate.rst

    Sampler
    BlockSampler
    NeighborSampler
    MultiLayerFullNeighborSampler
    ClusterGCNSampler
    ShaDowKHopSampler
    as_edge_prediction_sampler

.. _api-dataloading-negative-sampling:

Negative Samplers for Link Prediction
-------------------------------------
.. currentmodule:: dgl.dataloading.negative_sampler

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    Uniform
    PerSourceUniform
    GlobalUniform

Utility Class and Functions for Feature Prefetching
---------------------------------------------------
.. currentmodule:: dgl.dataloading.base

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    set_node_lazy_features
    set_edge_lazy_features
    set_src_lazy_features
    set_dst_lazy_features
    LazyFeature
