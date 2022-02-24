.. _apinn:

dgl.nn
==========

PyTorch
----------------------------------------

Conv Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dgl.nn.pytorch.conv
.. automodule:: dgl.nn.pytorch.conv

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    DenseGraphConv
    DenseSAGEConv
    DenseChebConv

Global Pooling Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dgl.nn.pytorch.glob
.. automodule:: dgl.nn.pytorch.glob

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dgl.nn.pytorch.link
.. automodule:: dgl.nn.pytorch.link

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    EdgePredictor
    TransE
    TransR

Heterogeneous Learning Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dgl.nn.pytorch
.. automodule:: dgl.nn.pytorch

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    HeteroGraphConv
    HeteroLinear
    HeteroEmbedding
    TypedLinear

Utility Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

TensorFlow
----------------------------------------

Conv Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dgl.nn.tensorflow.conv
.. automodule:: dgl.nn.tensorflow.conv

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    GraphConv
    RelGraphConv
    GATConv
    SAGEConv
    ChebConv
    SGConv
    APPNPConv
    GINConv

Global Pooling Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dgl.nn.tensorflow.glob
.. automodule:: dgl.nn.tensorflow.glob

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    SumPooling
    AvgPooling
    MaxPooling
    SortPooling
    GlobalAttentionPooling

Heterogeneous Learning Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dgl.nn.tensorflow
.. automodule:: dgl.nn.tensorflow

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    HeteroGraphConv

MXNet
----------------------------------------

Conv Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dgl.nn.mxnet.conv
.. automodule:: dgl.nn.mxnet.conv

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    GraphConv
    RelGraphConv
    TAGConv
    GATConv
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

Dense Conv Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    DenseGraphConv
    DenseSAGEConv
    DenseChebConv

Global Pooling Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dgl.nn.mxnet.glob
.. automodule:: dgl.nn.mxnet.glob

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    SumPooling
    AvgPooling
    MaxPooling
    SortPooling
    GlobalAttentionPooling
    Set2Set

Heterogeneous Learning Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: dgl.nn.mxnet
.. automodule:: dgl.nn.mxnet

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    HeteroGraphConv

Utility Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    ~utils.Sequential
