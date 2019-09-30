.. _apinn-mxnet:

NN Modules (MXNet)
===================

.. contents:: Contents
    :local:

We welcome your contribution! If you want a model to be implemented in DGL as a NN module,
please `create an issue <https://github.com/dmlc/dgl/issues>`_ started with "[Feature Request] NN Module XXXModel".

If you want to contribute a NN module, please `create a pull request <https://github.com/dmlc/dgl/pulls>`_ started
with "[NN] XXXModel in MXNet NN Modules" and our team member would review this PR.

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


Utility Modules
----------------------------------------

Edge Softmax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dgl.nn.mxnet.softmax
    :members: edge_softmax
