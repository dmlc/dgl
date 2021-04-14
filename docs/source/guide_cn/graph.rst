.. _guide_cn-graph:

第1章：图
=============

:ref:`(English Version)<guide-graph>`

图表示实体(节点)和它们的关系(边)，其中节点和边可以是有类型的 (例如，``"用户"`` 和 ``"物品"`` 是两种不同类型的节点)。
DGL通过其核心数据结构  :class:`~dgl.DGLGraph` 提供了一个以图为中心的编程抽象。 :class:`~dgl.DGLGraph` 提供了接口以处理图的结构、节点/边
的特征，以及使用这些组件可以执行的计算。


本章路线图
--------------

本章首先简要介绍了图的定义（见1.1节），然后介绍了一些 :class:`~dgl.DGLGraph` 相关的核心概念：

* :ref:`guide_cn-graph-basic`
* :ref:`guide_cn-graph-graphs-nodes-edges`
* :ref:`guide_cn-graph-feature`
* :ref:`guide_cn-graph-external`
* :ref:`guide_cn-graph-heterogeneous`
* :ref:`guide_cn-graph-gpu`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    graph-basic
    graph-graphs-nodes-edges
    graph-feature
    graph-external
    graph-heterogeneous
    graph-gpu
