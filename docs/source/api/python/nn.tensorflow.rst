.. _apinn-tensorflow:

NN Modules (Tensorflow)
====================

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
    
ChebConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.conv.ChebConv
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

Heterogeneous Graph Convolution Module
----------------------------------------

HeteroGraphConv
~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.tensorflow.HeteroGraphConv
   :members:
   :show-inheritance:
