.. _apinn-mxnet:

NN Modules (MXNet)
===================

Conv Layers
----------------------------------------

.. automodule:: dgl.nn.mxnet.conv

GraphConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.GraphConv
    :members: weight, bias, forward
    :show-inheritance:

RelGraphConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.RelGraphConv
    :members: forward
    :show-inheritance:

TAGConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.TAGConv
    :members: forward
    :show-inheritance:

GATConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.GATConv
    :members: forward
    :show-inheritance:

EdgeConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.EdgeConv
    :members: forward
    :show-inheritance:

SAGEConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.SAGEConv
    :members: forward
    :show-inheritance:

SGConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.SGConv
    :members: forward
    :show-inheritance:

APPNPConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.APPNPConv
    :members: forward
    :show-inheritance:

GINConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.GINConv
    :members: forward
    :show-inheritance:

GatedGraphConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.GatedGraphConv
    :members: forward
    :show-inheritance:

GMMConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.GMMConv
    :members: forward
    :show-inheritance:

ChebConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.ChebConv
    :members: forward
    :show-inheritance:

AGNNConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.AGNNConv
    :members: forward
    :show-inheritance:

NNConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.NNConv
    :members: forward
    :show-inheritance:

Dense Conv Layers
----------------------------------------

DenseGraphConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.DenseGraphConv
    :members: forward
    :show-inheritance:

DenseSAGEConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.DenseSAGEConv
    :members: forward
    :show-inheritance:

DenseChebConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.conv.DenseChebConv
    :members: forward
    :show-inheritance:

Global Pooling Layers
----------------------------------------

.. automodule:: dgl.nn.mxnet.glob

SumPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.glob.SumPooling
    :members:
    :show-inheritance:

AvgPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.glob.AvgPooling
    :members:
    :show-inheritance:

MaxPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.glob.MaxPooling
    :members:
    :show-inheritance:

SortPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.glob.SortPooling
    :members:
    :show-inheritance:

GlobalAttentionPooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.glob.GlobalAttentionPooling
    :members:
    :show-inheritance:

Set2Set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.glob.Set2Set
    :members:
    :show-inheritance:

Heterogeneous Graph Convolution Module
----------------------------------------

HeteroGraphConv
~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.HeteroGraphConv
   :members:
   :show-inheritance:

Utility Modules
----------------------------------------

Sequential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.mxnet.utils.Sequential
    :members:
    :show-inheritance:
