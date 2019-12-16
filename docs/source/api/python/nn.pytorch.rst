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

GraphConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.GraphConv
    :members: weight, bias, forward, reset_parameters
    :show-inheritance:
    
RelGraphConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.RelGraphConv
    :members: forward
    :show-inheritance:
    
TAGConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.TAGConv
    :members: forward
    :show-inheritance:
    
GATConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.GATConv
    :members: forward
    :show-inheritance:
    
EdgeConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.EdgeConv
    :members: forward
    :show-inheritance:
    
SAGEConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.SAGEConv
    :members: forward
    :show-inheritance:
    
SGConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.SGConv
    :members: forward
    :show-inheritance:
    
APPNPConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.APPNPConv
    :members: forward
    :show-inheritance:
    
GINConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.GINConv
    :members: forward
    :show-inheritance:
    
GatedGraphConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.GatedGraphConv
    :members: forward
    :show-inheritance:
    
GMMConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.GMMConv
    :members: forward
    :show-inheritance:
    
ChebConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.ChebConv
    :members: forward
    :show-inheritance:
    
AGNNConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.AGNNConv
    :members: forward
    :show-inheritance:
    
NNConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.NNConv
    :members: forward
    :show-inheritance:

AtomicConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.AtomicConv
    :members: forward
    :show-inheritance:
    
Dense Conv Layers
----------------------------------------

DenseGraphConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.DenseGraphConv
    :members: forward
    :show-inheritance:
    
DenseSAGEConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.DenseSAGEConv
    :members: forward
    :show-inheritance:
    
DenseChebConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.DenseChebConv
    :members: forward
    :show-inheritance:
    
Global Pooling Layers 
----------------------------------------

.. automodule:: dgl.nn.pytorch.glob

SumPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.glob.SumPooling
    :members:
    :show-inheritance:
    
AvgPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.glob.AvgPooling
    :members:
    :show-inheritance:
    
MaxPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.glob.MaxPooling
    :members:
    :show-inheritance:
    
SortPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.glob.SortPooling
    :members:
    :show-inheritance:
    
GlobalAttentionPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.glob.GlobalAttentionPooling
    :members:
    :show-inheritance:
    
Set2Set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.glob.Set2Set
    :members: forward
    :show-inheritance:
    
SetTransformerEncoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.glob.SetTransformerEncoder
    :members:
    :show-inheritance:
    
SetTransformerDecoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.glob.SetTransformerDecoder
    :members:
    :show-inheritance:
    
Utility Modules
----------------------------------------

KNNGraph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.factory.KNNGraph
    :members:
    :show-inheritance:
    
SegmentedKNNGraph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.factory.SegmentedKNNGraph
    :members:
    :show-inheritance:
    
Edge Softmax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dgl.nn.pytorch.softmax
    :members: edge_softmax
