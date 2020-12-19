.. _guide_cn-minibatch:

第6章：在大图上的随机（批次）训练
=======================================================

:ref:`(English Version) <guide-minibatch>`

If we have a massive graph with, say, millions or even billions of nodes
or edges, usually full-graph training as described in
:ref:`guide-training`
would not work. Consider an :math:`L`-layer graph convolutional network
with hidden state size :math:`H` running on an :math:`N`-node graph.
Storing the intermediate hidden states requires :math:`O(NLH)` memory,
easily exceeding one GPU’s capacity with large :math:`N`.

如果用户有一个包含数百万甚至数十亿个节点或边的大图，通常无法进行
:ref:`guide_cn-training`
中所述的全图训练。考虑在一个有N个节点的图上运行的、隐藏状态大小为H的L层图卷积网络，
存储隐藏层中间结果需要O(NLH)的内存空间，当N较大时，这很容易超过一块GPU的显存限制。

This section provides a way to perform stochastic minibatch training,
where we do not have to fit the feature of all the nodes into GPU.

这一节阐述了一种进行随机小批次训练的方法，使用户不用一次性把所有节点特征拷贝到GPU上。

Overview of Neighborhood Sampling Approaches

邻居采样方法概述
--------------------------------------------

Neighborhood sampling methods generally work as the following. For each
gradient descent step, we select a minibatch of nodes whose final
representations at the :math:`L`-th layer are to be computed. We then
take all or some of their neighbors at the :math:`L-1` layer. This
process continues until we reach the input. This iterative process
builds the dependency graph starting from the output and working
backwards to the input, as the figure below shows:

邻居节点采样的工作流程通常如下：每次梯度下降，选择一个小批次的图节点，其最终表示将在网络的第L层进行计算，
然后在网络的第L-1层选择该批次节点的全部或部分邻居节点。重复这个过程，直到到达输入层。
这个迭代过程将构建依赖关系图，从输出开始，一直到输入，如下图所示：

.. figure:: https://data.dgl.ai/asset/image/guide_6_0_0.png
   :alt: Imgur

With this, one can save the workload and computation resources for
training a GNN on a large graph.

该方法能节省在大图上训练图神经网络的开销和计算资源。

DGL provides a few neighborhood samplers and a pipeline for training a
GNN with neighborhood sampling, as well as ways to customize your
sampling strategies.

DGL实现了一些邻居节点采样的方法和使用邻居节点采样训练图神经网络的管道，也支持让用户自定义采样策略。

Roadmap

本章路线图
-----------

The chapter starts with sections for training GNNs stochastically under
different scenarios.

本章前半部分介绍了如何在不同场景下随机训练图神经网络。

* :ref:`guide_cn-minibatch-node-classification-sampler`
* :ref:`guide_cn-minibatch-edge-classification-sampler`
* :ref:`guide_cn-minibatch-link-classification-sampler`

The remaining sections cover more advanced topics, suitable for those who
wish to develop new sampling algorithms, new GNN modules compatible with
mini-batch training and understand how evaluation and inference can be
conducted in mini-batches.

在随后的小节里介绍了更多高级主题，面向那些想要开发新的采样算、
想要实现与小批次训练兼容的图神经网络模块以及了解如何在小批次数据上进行评估和推理模型的用户。

* :ref:`guide_cn-minibatch-customizing-neighborhood-sampler`
* :ref:`guide_cn-minibatch-custom-gnn-module`
* :ref:`guide_cn-minibatch-inference`


.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    minibatch-node
    minibatch-edge
    minibatch-link
    minibatch-custom-sampler
    minibatch-nn
    minibatch-inference
