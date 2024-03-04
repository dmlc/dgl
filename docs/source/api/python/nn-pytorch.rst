.. _apinn-pytorch:

dgl.nn (PyTorch)
================

Conv Layers
----------------------------------------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    ~dgl.nn.pytorch.conv.GraphConv
    ~dgl.nn.pytorch.conv.EdgeWeightNorm
    ~dgl.nn.pytorch.conv.RelGraphConv
    ~dgl.nn.pytorch.conv.TAGConv
    ~dgl.nn.pytorch.conv.GATConv
    ~dgl.nn.pytorch.conv.GATv2Conv
    ~dgl.nn.pytorch.conv.EGATConv
    ~dgl.nn.pytorch.conv.EdgeGATConv
    ~dgl.nn.pytorch.conv.EdgeConv
    ~dgl.nn.pytorch.conv.SAGEConv
    ~dgl.nn.pytorch.conv.SGConv
    ~dgl.nn.pytorch.conv.APPNPConv
    ~dgl.nn.pytorch.conv.GINConv
    ~dgl.nn.pytorch.conv.GINEConv
    ~dgl.nn.pytorch.conv.GatedGraphConv
    ~dgl.nn.pytorch.conv.GatedGCNConv
    ~dgl.nn.pytorch.conv.GMMConv
    ~dgl.nn.pytorch.conv.ChebConv
    ~dgl.nn.pytorch.conv.AGNNConv
    ~dgl.nn.pytorch.conv.NNConv
    ~dgl.nn.pytorch.conv.AtomicConv
    ~dgl.nn.pytorch.conv.CFConv
    ~dgl.nn.pytorch.conv.DotGatConv
    ~dgl.nn.pytorch.conv.TWIRLSConv
    ~dgl.nn.pytorch.conv.TWIRLSUnfoldingAndAttention
    ~dgl.nn.pytorch.conv.GCN2Conv
    ~dgl.nn.pytorch.conv.HGTConv
    ~dgl.nn.pytorch.conv.GroupRevRes
    ~dgl.nn.pytorch.conv.EGNNConv
    ~dgl.nn.pytorch.conv.PNAConv
    ~dgl.nn.pytorch.conv.DGNConv

CuGraph Conv Layers
----------------------------------------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    ~dgl.nn.pytorch.conv.CuGraphRelGraphConv
    ~dgl.nn.pytorch.conv.CuGraphGATConv
    ~dgl.nn.pytorch.conv.CuGraphSAGEConv

Dense Conv Layers
----------------------------------------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    ~dgl.nn.pytorch.conv.DenseGraphConv
    ~dgl.nn.pytorch.conv.DenseSAGEConv
    ~dgl.nn.pytorch.conv.DenseChebConv

Global Pooling Layers
----------------------------------------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    ~dgl.nn.pytorch.glob.SumPooling
    ~dgl.nn.pytorch.glob.AvgPooling
    ~dgl.nn.pytorch.glob.MaxPooling
    ~dgl.nn.pytorch.glob.SortPooling
    ~dgl.nn.pytorch.glob.WeightAndSum
    ~dgl.nn.pytorch.glob.GlobalAttentionPooling
    ~dgl.nn.pytorch.glob.Set2Set
    ~dgl.nn.pytorch.glob.SetTransformerEncoder
    ~dgl.nn.pytorch.glob.SetTransformerDecoder

Score Modules for Link Prediction and Knowledge Graph Completion
----------------------------------------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    ~dgl.nn.pytorch.link.EdgePredictor
    ~dgl.nn.pytorch.link.TransE
    ~dgl.nn.pytorch.link.TransR

Heterogeneous Learning Modules
----------------------------------------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    ~dgl.nn.pytorch.HeteroGraphConv
    ~dgl.nn.pytorch.HeteroLinear
    ~dgl.nn.pytorch.HeteroEmbedding
    ~dgl.nn.pytorch.TypedLinear

Utility Modules
----------------------------------------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    ~dgl.nn.pytorch.utils.Sequential
    ~dgl.nn.pytorch.utils.WeightBasis
    ~dgl.nn.pytorch.factory.KNNGraph
    ~dgl.nn.pytorch.factory.SegmentedKNNGraph
    ~dgl.nn.pytorch.factory.RadiusGraph
    ~dgl.nn.pytorch.utils.JumpingKnowledge
    ~dgl.nn.pytorch.sparse_emb.NodeEmbedding
    ~dgl.nn.pytorch.explain.GNNExplainer
    ~dgl.nn.pytorch.explain.HeteroGNNExplainer
    ~dgl.nn.pytorch.explain.SubgraphX
    ~dgl.nn.pytorch.explain.HeteroSubgraphX
    ~dgl.nn.pytorch.explain.PGExplainer
    ~dgl.nn.pytorch.explain.HeteroPGExplainer
    ~dgl.nn.pytorch.utils.LabelPropagation
    ~dgl.nn.pytorch.utils.LaplacianPosEnc

Network Embedding Modules
----------------------------------------

.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    ~dgl.nn.pytorch.DeepWalk
    ~dgl.nn.pytorch.MetaPath2Vec

Utility Modules for Graph Transformer
----------------------------------------
.. autosummary::
    :toctree: ../../generated/
    :nosignatures:
    :template: classtemplate.rst

    ~dgl.nn.pytorch.gt.DegreeEncoder
    ~dgl.nn.pytorch.gt.LapPosEncoder
    ~dgl.nn.pytorch.gt.PathEncoder
    ~dgl.nn.pytorch.gt.SpatialEncoder
    ~dgl.nn.pytorch.gt.SpatialEncoder3d
    ~dgl.nn.pytorch.gt.BiasedMHA
    ~dgl.nn.pytorch.gt.GraphormerLayer
    ~dgl.nn.pytorch.gt.EGTLayer
