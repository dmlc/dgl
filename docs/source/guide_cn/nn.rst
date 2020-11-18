.. _guide_cn-nn:

第3章：构建图神经网络（GNN）模块
===================================

:ref:`(English Version) <guide-nn>`

DGL NN模块是用户构建GNN模型的基本模块。根据DGL所使用的后端深度神经网络框架，
DGL NN模块的父类取决于后端所使用的深度神经网络框架。对于PyTorch后端，
它应该继承 `PyTorch的NN模块 <https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/module.html>`__；对于MXNet后端，它应该继承
`MXNet Gluon的NN块 <http://mxnet.incubator.apache.org/versions/1.6/api/python/docs/api/gluon/nn/index.html>`__；
对于TensorFlow后端，它应该继承 `Tensorflow的Keras层 <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__。
在DGL NN模块中，构造函数中的参数注册和前向传播函数中使用的张量操作与后端框架一样。这种方式使得DGL的代码可以无缝嵌入到后端框架的代码中。
DGL和这些深度神经网络框架的主要差异是其独有的消息传递操作。

DGL已经集成了很多常用的 :ref:`apinn-pytorch-conv`、 :ref:`apinn-pytorch-dense-conv`、
:ref:`apinn-pytorch-pooling` 和 :ref:`apinn-pytorch-util`。欢迎给DGL贡献更多的模块！

本章将使用PyTorch作为后端，用 :class:`~dgl.nn.pytorch.conv.SAGEConv` 作为例子来介绍如何构建用户自己的DGL NN模块。

本章路线图
------------

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
