.. _apinn-tensorflow:

NN Modules (Tensorflow)
====================

.. contents:: Contents
    :local:

We welcome your contribution! If you want a model to be implemented in DGL as a NN module,
please `create an issue <https://github.com/dmlc/dgl/issues>`_ started with "[Feature Request] NN Module XXXModel".

If you want to contribute a NN module, please `create a pull request <https://github.com/dmlc/dgl/pulls>`_ started
with "[NN] XXXModel in tensorflow NN Modules" and our team member would review this PR.

Conv Layers 
----------------------------------------

.. automodule:: dgl.nn.tensorflow.conv

GraphConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.GraphConv
    :members: weight, bias, forward, reset_parameters
    :show-inheritance:
    
RelGraphConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.RelGraphConv
    :members: forward
    :show-inheritance:
    
TAGConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.TAGConv
    :members: forward
    :show-inheritance:
    
GATConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.GATConv
    :members: forward
    :show-inheritance:
    
EdgeConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.EdgeConv
    :members: forward
    :show-inheritance:
    
SAGEConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.SAGEConv
    :members: forward
    :show-inheritance:
    
SGConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.SGConv
    :members: forward
    :show-inheritance:
    
APPNPConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.APPNPConv
    :members: forward
    :show-inheritance:
    
GINConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.GINConv
    :members: forward
    :show-inheritance:
    
GatedGraphConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.GatedGraphConv
    :members: forward
    :show-inheritance:
    
GMMConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.GMMConv
    :members: forward
    :show-inheritance:
    
ChebConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.ChebConv
    :members: forward
    :show-inheritance:
    
AGNNConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.AGNNConv
    :members: forward
    :show-inheritance:
    
NNConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.NNConv
    :members: forward
    :show-inheritance:

AtomicConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.AtomicConv
    :members: forward
    :show-inheritance:
    
Dense Conv Layers
----------------------------------------

DenseGraphConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.DenseGraphConv
    :members: forward
    :show-inheritance:
    
DenseSAGEConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.DenseSAGEConv
    :members: forward
    :show-inheritance:
    
DenseChebConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.DenseChebConv
    :members: forward
    :show-inheritance:
    
Global Pooling Layers 
----------------------------------------

.. automodule:: dgl.nn.tensorflow.glob

SumPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.glob.SumPooling
    :members:
    :show-inheritance:
    
AvgPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.glob.AvgPooling
    :members:
    :show-inheritance:
    
MaxPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.glob.MaxPooling
    :members:
    :show-inheritance:
    
SortPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.glob.SortPooling
    :members:
    :show-inheritance:
    
GlobalAttentionPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.glob.GlobalAttentionPooling
    :members:
    :show-inheritance:
    
Set2Set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.glob.Set2Set
    :members: forward
    :show-inheritance:
    
SetTransformerEncoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.glob.SetTransformerEncoder
    :members:
    :show-inheritance:
    
SetTransformerDecoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.glob.SetTransformerDecoder
    :members:
    :show-inheritance:
    
Utility Modules
----------------------------------------

Sequential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.utils.Sequential
    :members:
    :show-inheritance:

KNNGraph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.factory.KNNGraph
    :members:
    :show-inheritance:
    
SegmentedKNNGraph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.factory.SegmentedKNNGraph
    :members:
    :show-inheritance:
    
Edge Softmax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dgl.nn.tensorflow.softmax
    :members: edge_softmax
