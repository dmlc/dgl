.. _apinn-pytorch:

dgl.nn (PyTorch)
================

Conv Layers
----------------------------------------

.. currentmodule:: dgl.nn.pytorch.conv

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    GraphConv
    EdgeWeightNorm
    RelGraphConv
    TAGConv
    GATConv
    GATv2Conv
    EGATConv
    EdgeConv
    SAGEConv
    SGConv
    APPNPConv
    GINConv
    GatedGraphConv
    GMMConv
    ChebConv
    AGNNConv
    NNConv
    AtomicConv
    CFConv
    DotGatConv
    TWIRLSConv
    TWIRLSUnfoldingAndAttention
    GCN2Conv

Dense Conv Layers
----------------------------------------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    DenseGraphConv
    DenseSAGEConv
    DenseChebConv

Global Pooling Layers
----------------------------------------

.. currentmodule:: dgl.nn.pytorch.glob

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    SumPooling
    AvgPooling
    MaxPooling
    SortPooling
    WeightAndSum
    GlobalAttentionPooling
    Set2Set
    SetTransformerEncoder
    SetTransformerDecoder

Score Modules for Link Prediction and Knowledge Graph Completion
----------------------------------------

.. currentmodule:: dgl.nn.pytorch.link

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    EdgePredictor
    TransE
    TransR

Heterogeneous Learning Modules
----------------------------------------

.. currentmodule:: dgl.nn.pytorch

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    HeteroGraphConv
    HeteroLinear
    HeteroEmbedding
    TypedLinear

Utility Modules
----------------------------------------

.. currentmodule:: dgl.nn.pytorch

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    ~utils.Sequential
    ~utils.WeightBasis
    ~factory.KNNGraph
    ~factory.SegmentedKNNGraph
    ~utils.JumpingKnowledge
    ~sparse_emb.NodeEmbedding
    ~explain.GNNExplainer
