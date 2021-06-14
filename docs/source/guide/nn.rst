.. _guide-nn:

Chapter 3: Building GNN Modules
===============================

:ref:`(中文版) <guide_cn-nn>`

DGL NN module consists of building blocks for GNN models. An NN module inherits
from `Pytorch’s NN Module <https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/module.html>`__, `MXNet Gluon’s NN Block  <http://mxnet.incubator.apache.org/versions/1.6/api/python/docs/api/gluon/nn/index.html>`__ and `TensorFlow’s Keras
Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__, depending on the DNN framework backend in use. In a DGL NN
module, the parameter registration in construction function and tensor
operation in forward function are the same with the backend framework.
In this way, DGL code can be seamlessly integrated into the backend
framework code. The major difference lies in the message passing
operations that are unique in DGL.

DGL has integrated many commonly used
:ref:`apinn-pytorch-conv`, :ref:`apinn-pytorch-dense-conv`, :ref:`apinn-pytorch-pooling`,
and
:ref:`apinn-pytorch-util`. We welcome your contribution!

This chapter takes :class:`~dgl.nn.pytorch.conv.SAGEConv` with Pytorch backend as an example
to introduce how to build a custom DGL NN Module.

Roadmap
-------

* :ref:`guide-nn-construction`
* :ref:`guide-nn-forward`
* :ref:`guide-nn-heterograph`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    nn-construction
    nn-forward
    nn-heterograph
