.. _api-dataloading:

dgl.dataloading
=================================

.. currentmodule:: dgl.dataloading

The ``dgl.dataloading`` package provides two primitives to compose a data pipeline
for loading from graph data. ``Sampler`` represents algorithms
to generate subgraph samples from the original graph, and ``DataLoader``
represents the iterable over these samples.

DGL provides a number of built-in samplers that subclass :class:`~dgl.dataloading.Sampler`.
Creating new samplers follow the same paradigm. Read our user guide chapter
:ref:`guide-minibatch` for more examples and explanations.

The entire package only works for PyTorch backend.

DataLoaders
-----------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    DataLoader
    GraphDataLoader

.. _api-dataloading-neighbor-sampling:

Samplers
--------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    Sampler
    NeighborSampler
    LaborSampler
    MultiLayerFullNeighborSampler
    ClusterGCNSampler
    ShaDowKHopSampler
    SAINTSampler

Sampler Transformations
-----------------------

.. autosummary::
    :toctree: ../../generated/

    as_edge_prediction_sampler
    BlockSampler

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
