.. _guide-graph:

Chapter 1: Graph
======================

:ref:`(中文版)<guide_cn-graph>`

Graphs express entities (nodes) along with their relations (edges), and both nodes and
edges can be typed (e.g., ``"user"`` and ``"item"`` are two different types of nodes). DGL provides a
graph-centric programming abstraction with its core data structure -- :class:`~dgl.DGLGraph`. :class:`~dgl.DGLGraph`
provides its interface to handle a graph's structure, its node/edge features, and the resulting
computations that can be performed using these components.

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
