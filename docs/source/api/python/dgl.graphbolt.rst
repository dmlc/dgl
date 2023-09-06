.. _apibackend:

🆕 dgl.graphbolt
=================================

.. currentmodule:: dgl.graphbolt

`dgl.graphbolt` is a dataloading framework for GNN that provides well-defined APIs for each stage of the data pipeline and multiple standard implementations.

APIs
-------------------------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    Dataset
    Task
    ItemSet
    ItemSetDict
    ItemSampler
    NegativeSampler
    SubgraphSampler
    Feature
    FeatureStore
    FeatureFetcher
    CopyTo

DataLoaders
-----------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    SingleProcessDataLoader
    MultiProcessDataLoader
