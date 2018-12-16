.. DGL documentation master file, created by
   sphinx-quickstart on Fri Oct  5 14:18:01 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview of DGL
===============

Deep Graph Library (DGL) is a Python package built for easy implementation of
graph neural network model family, on top of existing DL frameworks (e.g.
Pytorch, MXNet, Gluon etc.).

DGL reduces the implementation of graph neural networks into declaring a set
of *functions* (or *modules* in PyTorch terminology).  In addition, DGL
provides:

* Versatile controls over message passing, ranging from low-level operations
  such as sending along selected edges and receiving on specific nodes, to
  high-level control such as graph-wide feature updates.
* Transparent speed optimization with automatic batching of computations and
  sparse matrix multiplication.
* Seamless integration with existing deep learning frameworks.
* Easy and friendly interfaces for node/edge feature access and graph
  structure manipulation.
* Good scalability to graphs with tens of millions of vertices.

To begin with, we have prototyped 10 models across various domains:
semi-supervised learning on graphs (with potentially billions of nodes/edges),
generative models on graphs, (previously) difficult-to-parallelize tree-based
models like TreeLSTM, etc. We also implement some conventional models in DGL
from a new graphical perspective yielding simplicity.

Relationship of DGL to other frameworks
---------------------------------------
DGL is designed to be compatible and agnostic to the existing tensor
frameworks. It provides a backend adapter interface that allows easy porting
to other tensor-based, autograd-enabled frameworks. Currently, our prototype
works with MXNet/Gluon and PyTorch.

Free software
-------------
DGL is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/dmlc/dgl>`_ and checkout our `contribution guidelines <https://github.com/dmlc/dgl/blob/master/CONTRIBUTING.md>`_.

History
-------
Prototype of DGL started in early Spring, 2018, at NYU Shanghai by Prof. Zheng
Zhang and Quan Gan. Serious development began when Minjie, Lingfan and Prof
Jinyang Li from NYU's system group joined, flanked by a team of student
volunteers at NYU Shanghai, Fudan and other universities (Yu, Zihao, Murphy,
Allen, Qipeng, Qi, Hao), as well as early adopters at the CILVR lab (Jake
Zhao). Development accelerated when AWS MXNet Science team joined force, with
Da Zheng, Alex Smola, Haibin Lin, Chao Ma and a number of others. For full
credit, see `here <https://www.dgl.ai/ack>`_.

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :glob:

   install/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :glob:

   tutorials/basics/index
   tutorials/models/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :glob:

   api/python/index

.. toctree::
   :maxdepth: 1
   :glob:

   faq
   env_var

Index
-----
* :ref:`genindex`
