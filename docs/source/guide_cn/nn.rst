.. _guide_cn-nn:

第3章：构建图神经网络（GNN）模块
===================================

:ref:`(English Version) <guide-nn>`

DGL NN module consists of building blocks for GNN models. An NN module inherits
from `Pytorch’s NN Module <https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/module.html>`__, `MXNet Gluon’s NN Block <http://mxnet.incubator.apache.org/versions/1.6/api/python/docs/api/gluon/nn/index.html>`__ and `TensorFlow’s Keras
Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__, depending on the DNN framework backend in use. In a DGL NN
module, the parameter registration in construction function and tensor
operation in forward function are the same with the backend framework.
In this way, DGL code can be seamlessly integrated into the backend
framework code. The major difference lies in the message passing
operations that are unique in DGL.

DGL NN模块是用户构建GNN模型的基本模块。根据后端使用的深度神经网络框架，
它继承了 `PyTorch的NN模块 <https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/module.html>`__，
`MXNet Gluon的NN Block <http://mxnet.incubator.apache.org/versions/1.6/api/python/docs/api/gluon/nn/index.html>`__ 和
`Tensorflow的Keras Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__。
在DGL NN模块中，构造函数中注册的参数和前向传播函数中使用的张量操作与后端框架一样。这种方式使得DGL的代码可以无缝嵌入到后端框架的代码中。
DGL和这些深度神经网络框架的主要差异是其独有的消息传递操作。

DGL has integrated many commonly used
:ref:`apinn-pytorch-conv`, :ref:`apinn-pytorch-dense-conv`, :ref:`apinn-pytorch-pooling`,
and
:ref:`apinn-pytorch-util`. We welcome your contribution!

DGL已经集成了很多常用的 :ref:`apinn-pytorch-conv` 层, :ref:`apinn-pytorch-dense-conv` 层,
:ref:`apinn-pytorch-pooling` 层和 :ref:`apinn-pytorch-util` 模块。欢迎给DGL贡献更多模块！

This chapter takes :class:`~dgl.nn.pytorch.conv.SAGEConv` with Pytorch backend as an example
to introduce how to build a custom DGL NN Module.

本章内容将使用PyTorch作为后端，用 :class:`~dgl.nn.pytorch.conv.SAGEConv` 作为例子来介绍如何构建用户自己的DGL NN模块。

Roadmap
-------

* :ref:`guide_cn-nn-construction`
* :ref:`guide_cn-nn-forward`
* :ref:`guide_cn-nn-heterograph`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    nn-construction
    nn-forward
    nn-heterograph
