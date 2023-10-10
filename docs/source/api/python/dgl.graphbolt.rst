.. _apibackend:

🆕 dgl.graphbolt
=================================

.. currentmodule:: dgl.graphbolt

**dgl.graphbolt** is a dataloading framework for GNN that provides well-defined APIs for each stage of the data pipeline and multiple standard implementations.

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
    DistributedItemSampler
    NegativeSampler
    SubgraphSampler
    SampledSubgraph
    SamplingGraph
    MiniBatch
    MiniBatchTransformer
    DGLMiniBatch
    DGLMiniBatchConverter
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

Standard Implementations
-------------------------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    OnDiskDataset
    BuiltinDataset
    OnDiskTask
    OnDiskMetaData
    CSCSamplingGraph
    UniformNegativeSampler
    NeighborSampler
    LayerNeighborSampler
    SampledSubgraphImpl
    BasicFeatureStore
    TorchBasedFeature
    TorchBasedFeatureStore
    GPUCachedFeature
