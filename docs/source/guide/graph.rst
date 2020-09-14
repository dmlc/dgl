.. _guide-graph:

Chapter 1: Graph
======================

Graphs express entities (nodes) along with their relations (edges), and both nodes and
edges can be typed (e.g., ``"user"`` and ``"item"`` are two different types of nodes). DGL provides a
graph-centric programming abstraction with its core data structure -- :class:`~dgl.DGLGraph`. :class:`~dgl.DGLGraph`
provides its interface to handle a graph's structure, its node/edge features, and the resulting
computations that can be performed using these components.


图表示实体(节点)和它们的关系(边)，其中节点和边可以是有类型的 (例如，``用户``和``物品``是两种不同类型的节点)。
DGL通过其核心数据结构:class:`~dgl.DGLGraph`提供了一个以图为中心的编程抽象。:class:`~dgl.DGLGraph`提供了接口来处理图的结构、节点/边
特征以及使用这些组件可以执行的。

Roadmap
-------

The chapter starts with a brief introduction to graph definitions in 1.1 and then introduces some core
concepts of :class:`~dgl.DGLGraph`:

* :ref:`guide-graph-basic`
* :ref:`guide-graph-graphs-nodes-edges`
* :ref:`guide-graph-feature`
* :ref:`guide-graph-external`
* :ref:`guide-graph-heterogeneous`
* :ref:`guide-graph-gpu`

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
