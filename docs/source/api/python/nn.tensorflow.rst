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
    
GATConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.GATConv
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
