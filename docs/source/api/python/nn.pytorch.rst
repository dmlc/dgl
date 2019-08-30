.. _apinn-pytorch:

NN Modules (PyTorch)
====================

.. contents:: Contents
    :local:

We welcome your contribution! If you want a model to be implemented in DGL as a NN module,
please `create an issue <https://github.com/dmlc/dgl/issues>`_ started with "[Feature Request] NN Module XXXModel".

If you want to contribute a NN module, please `create a pull request <https://github.com/dmlc/dgl/pulls>`_ started
with "[NN] XXXModel in PyTorch NN Modules" and our team member would review this PR.

Conv Layers 
----------------------------------------

.. automodule:: dgl.nn.pytorch.conv

.. autoclass:: dgl.nn.pytorch.conv.GraphConv
    :members: weight, bias, forward, reset_parameters
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.RelGraphConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.TAGConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.GATConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.EdgeConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.SAGEConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.SGConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.APPNPConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.GINConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.GatedGraphConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.GMMConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.ChebConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.AGNNConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.NNConv
    :members: forward
    :show-inheritance:
    
Dense Conv Layers
----------------------------------------

.. autoclass:: dgl.nn.pytorch.conv.DenseGraphConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.DenseSAGEConv
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.conv.DenseChebConv
    :members: forward
    :show-inheritance:
    
Global Pooling Layers 
----------------------------------------

.. automodule:: dgl.nn.pytorch.glob

.. autoclass:: dgl.nn.pytorch.glob.SumPooling
    :members:
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.glob.AvgPooling
    :members:
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.glob.MaxPooling
    :members:
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.glob.SortPooling
    :members:
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.glob.GlobalAttentionPooling
    :members:
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.glob.Set2Set
    :members: forward
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.glob.SetTransformerEncoder
    :members:
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.glob.SetTransformerDecoder
    :members:
    :show-inheritance:
    
Utility Modules
----------------------------------------

.. autoclass:: dgl.nn.pytorch.factory.KNNGraph
    :members:
    :show-inheritance:
    
.. autoclass:: dgl.nn.pytorch.factory.SegmentedKNNGraph
    :members:
    :show-inheritance:
    
.. automodule:: dgl.nn.pytorch.softmax
    :members: edge_softmax
    

