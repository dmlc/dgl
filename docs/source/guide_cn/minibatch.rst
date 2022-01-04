.. _guide_cn-minibatch:

第6章：在大图上的随机（批次）训练
=======================================================

:ref:`(English Version) <guide-minibatch>`

如果用户有包含数百万甚至数十亿个节点或边的大图，通常无法进行
:ref:`guide_cn-training`
中所述的全图训练。考虑在一个有 :math:`N` 个节点的图上运行的、隐层大小为 :math:`H` 的 :math:`L` 层图卷积网络，
存储隐层表示需要 :math:`O(NLH)` 的内存空间，当 :math:`N` 较大时，这很容易超过一块GPU的显存限制。

本章介绍了一种在大图上进行随机小批次训练的方法，可以让用户不用一次性把所有节点特征拷贝到GPU上。

邻居采样方法概述
--------------------------------------------

邻居节点采样的工作流程通常如下：每次梯度下降，选择一个小批次的图节点，
其最终表示将在神经网络的第 :math:`L` 层进行计算，然后在网络的第 :math:`L-1` 层选择该批次节点的全部或部分邻居节点。
重复这个过程，直到到达输入层。这个迭代过程会构建计算的依赖关系图，从输出开始，一直到输入，如下图所示：

.. figure:: https://data.dgl.ai/asset/image/guide_6_0_0.png
   :alt: Imgur

该方法能节省在大图上训练图神经网络的开销和计算资源。

DGL实现了一些邻居节点采样的方法和使用邻居节点采样训练图神经网络的管道，同时也支持让用户自定义采样策略。

本章路线图
-----------

本章的前半部分介绍了不同场景下如何进行随机训练的方法。

* :ref:`guide_cn-minibatch-node-classification-sampler`
* :ref:`guide_cn-minibatch-edge-classification-sampler`
* :ref:`guide_cn-minibatch-link-classification-sampler`

本章余下的小节介绍了更多的高级主题，面向那些想要开发新的采样算法、
想要实现与小批次训练兼容的图神经网络模块、以及想要了解如何在小批次数据上进行评估和推理模型的用户。

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
