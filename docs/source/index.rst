.. DGL documentation master file, created by
   sphinx-quickstart on Fri Oct  5 14:18:01 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Deep Graph Library Tutorials and Documentation
=========================================================

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :hidden:
   :glob:

   install/index
   tutorials/blitz/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced Materials
   :hidden:
   :titlesonly:
   :glob:

   stochastic_training/index
   guide/index
   guide_cn/index
   guide_ko/index
   graphtransformer/index
   notebooks/sparse/index
   tutorials/cpu/index
   tutorials/multi/index
   tutorials/dist/index
   tutorials/models/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:
   :glob:

   api/python/dgl
   api/python/dgl.data
   api/python/dgl.dataloading
   api/python/dgl.DGLGraph
   api/python/dgl.distributed
   api/python/dgl.function
   api/python/dgl.geometry
   api/python/dgl.graphbolt
   api/python/nn-pytorch
   api/python/nn.functional
   api/python/dgl.ops
   api/python/dgl.optim
   api/python/dgl.sampling
   api/python/dgl.sparse_v0
   api/python/dgl.multiprocessing
   api/python/transforms
   api/python/udf

.. toctree::
   :maxdepth: 1
   :caption: Notes
   :hidden:
   :glob:

   contribute
   developer/ffi
   performance

.. toctree::
   :maxdepth: 1
   :caption: Misc
   :hidden:
   :glob:

   faq
   env_var
   resources


Deep Graph Library (DGL) is a Python package built for easy implementation of
graph neural network model family, on top of existing DL frameworks (currently
supporting PyTorch, MXNet and TensorFlow). It offers a versatile control of message passing,
speed optimization via auto-batching and highly tuned sparse matrix kernels,
and multi-GPU/CPU training to scale to graphs of hundreds of millions of
nodes and edges.

Getting Started
---------------

For absolute beginners, start with the :doc:`Blitz Introduction to DGL <tutorials/blitz/index>`.
It covers the basic concepts of common graph machine learning tasks and a step-by-step
on building Graph Neural Networks (GNNs) to solve them.

For acquainted users who wish to learn more advanced usage,

* `Learn DGL by examples <https://github.com/dmlc/dgl/tree/master/examples>`_.
* Read the :doc:`User Guide<guide/index>` (:doc:`中文版链接<guide_cn/index>`), which explains the concepts
  and usage of DGL in much more details.
* Go through the tutorials for :doc:`Stochastic Training of GNNs <notebooks/stochastic_training/index>`,
  which covers the basic steps for training GNNs on large graphs in mini-batches.
* :doc:`Study classical papers <tutorials/models/index>` on graph machine learning alongside DGL.
* Search for the usage of a specific API in the :doc:`API reference manual <api/python/index>`,
  which organizes all DGL APIs by their namespace.

Contribution
-------------
DGL is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/dmlc/dgl>`_ and check out our
:doc:`contribution guidelines <contribute>`.

Index
-----
* :ref:`genindex`
