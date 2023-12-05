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
    :template: graphbolt_classtemplate.rst

    DataLoader
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


Standard Implementations
-------------------------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: graphbolt_classtemplate.rst

    OnDiskDataset
    BuiltinDataset
    FusedCSCSamplingGraph
    UniformNegativeSampler
    NeighborSampler
    LayerNeighborSampler
    FusedSampledSubgraphImpl
    BasicFeatureStore
    TorchBasedFeature
    TorchBasedFeatureStore
    GPUCachedFeature
