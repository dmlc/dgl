.. _apinn-mxnet:

NN Modules (MXNet)
=================

Conv Layers 
----------------------------------------

.. automodule:: dgl.nn.mxnet.conv

.. autoclass:: dgl.nn.mxnet.conv.GraphConv
    :members: weight, bias, forward
    :show-inheritance:

.. autoclass:: dgl.nn.mxnet.conv.RelGraphConv
    :members: forward
    :show-inheritance:

.. autoclass:: dgl.nn.mxnet.conv.TAGConv
    :members: forward
    :show-inheritance:

Global Pooling Layers 
----------------------------------------

.. automodule:: dgl.nn.mxnet.glob
    :members:
    :show-inheritance:

Utility Modules
----------------------------------------

.. automodule:: dgl.nn.mxnet.softmax
    :members: edge_softmax
