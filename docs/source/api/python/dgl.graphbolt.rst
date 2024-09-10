.. _apibackend:

🆕 dgl.graphbolt
=================================

.. currentmodule:: dgl.graphbolt

**dgl.graphbolt** is a dataloading framework for GNNs that provides well-defined
APIs for each stage of the data pipeline and multiple standard implementations.

Dataset
-------

A dataset is a collection of graph structure data, feature data and tasks.

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: graphbolt_classtemplate.rst

    Dataset
    OnDiskDataset
    BuiltinDataset
    LegacyDataset
    Task

Graph
-----

A graph is a collection of nodes and edges. It can be a homogeneous graph or a
heterogeneous graph.

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: graphbolt_classtemplate.rst

    SamplingGraph
    FusedCSCSamplingGraph


Feature and FeatureStore
------------------------

A feature is a collection of data(tensor, array). A feature store is a
collection of features.

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: graphbolt_classtemplate.rst

    Feature
    FeatureStore
    BasicFeatureStore
    TorchBasedFeature
    TorchBasedFeatureStore
    DiskBasedFeature
    CPUCachedFeature
    GPUCachedFeature


DataLoader
----------

A dataloader is for iterating over a dataset and generate mini-batches.

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: graphbolt_classtemplate.rst

    DataLoader


ItemSet
-------

An item set is an iterable collection of items.

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: graphbolt_classtemplate.rst

    ItemSet
    HeteroItemSet


ItemSampler
-----------

An item sampler is for sampling items from an item set.

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: graphbolt_classtemplate.rst

    ItemSampler
    DistributedItemSampler


MiniBatch
---------

A mini-batch is a collection of sampled subgraphs and their corresponding
features. It is the basic unit for training a GNN model.

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: graphbolt_classtemplate.rst

    MiniBatch
    MiniBatchTransformer


NegativeSampler
---------------

A negative sampler is for sampling negative items from mini-batches.

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: graphbolt_classtemplate.rst

    NegativeSampler
    UniformNegativeSampler


SubgraphSampler
---------------

A subgraph sampler is for sampling subgraphs from a graph.

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: graphbolt_classtemplate.rst

    SubgraphSampler
    SampledSubgraph
    NeighborSampler
    LayerNeighborSampler
    TemporalNeighborSampler
    TemporalLayerNeighborSampler
    SampledSubgraphImpl
    FusedSampledSubgraphImpl
    InSubgraphSampler


FeatureFetcher
--------------

A feature fetcher is for fetching features from a feature store.

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: graphbolt_classtemplate.rst

    FeatureFetcher


CopyTo
------

This datapipe is for copying data to a device.

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: graphbolt_classtemplate.rst

    CopyTo


Utilities
---------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:

    cpu_cached_feature
    gpu_cached_feature
    fused_csc_sampling_graph
    load_from_shared_memory
    from_dglgraph
    etype_str_to_tuple
    etype_tuple_to_str
    isin
    seed
    index_select
    expand_indptr
    indptr_edge_ids
    add_reverse_edges
    exclude_seed_edges
    compact_csc_format
    unique_and_compact
    unique_and_compact_csc_formats
    numpy_save_aligned
