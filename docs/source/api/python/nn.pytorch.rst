.. _apinn-pytorch:

NN Modules (PyTorch)
====================

.. _apinn-pytorch-conv:

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

CFConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.CFConv
    :members: forward
    :show-inheritance:

DotGatConv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.conv.DotGatConv
    :members: forward
    :show-inheritance:

.. _apinn-pytorch-dense-conv:

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

.. _apinn-pytorch-pooling:

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

Heterogeneous Graph Convolution Module
----------------------------------------

HeteroGraphConv
~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.HeteroGraphConv
   :members:
   :show-inheritance:

.. _apinn-pytorch-util:

Utility Modules
----------------------------------------

Sequential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dgl.nn.pytorch.utils.Sequential
    :members:
    :show-inheritance:

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

