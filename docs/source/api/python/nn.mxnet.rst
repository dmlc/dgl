.. _apinn-mxnet:

NN Modules (MXNet)
===================

.. contents:: Contents
    :local:

We welcome your contribution! If you want a model to be implemented in DGL as a NN module,
create an issue at stared with "[Feature Request] NN Module XXXModel".

If you want to contribute a NN module, create a pull request started with "[NN] XXXModel in MXNet NN Modules"
and our team member would review this PR.

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
