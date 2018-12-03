.. DGL documentation master file, created by
   sphinx-quickstart on Fri Oct  5 14:18:01 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview of DGL
===============

Deep Graph Library (DGL) is a Python package that integrates existing DL
frameworks (e.g. Pytorch, MXNet, Gluon etc.) with learning on structural data
expressed as graphs.

DGL provides:

* Various levels of controls over message-passing, starting from the familiar
  mataphor of socket with send and recv among a pair of nodes, to update_all for
  a graph-wide update.
* A friendly, dictionary-like API to access feature tensors, using
  the mataphor of mailbox to collect and process messages.
* Convinent interfaces for querying and modifying graph structures.
* Seemless integration with existing DL frameworks in the form of user-defined functions (UDFs)
  for expressiveness and flexibility.
* Automatic batching of graphs (or sampled subgraphs) to explore and exploit
  maximum parallelism.
* Build-in operation to accelerate UDFs using sparse-matrix
  operation when possible.

We have prototyped altogether 10 different models,
all of them are ready to run out-of-box. The tasks range from semi-supervised
learning on graphs, scaling them with sampling techniques, generative models of
graphs, (previously) hard-to-parallelize tree-structured algorithms, and our
attempts to cast some classical models in light of graphs. Note that some of
the models are very new graph-based algorithms -- DGL’s goal is to facilitate
further growth of research in this area.

Relationship of DGL to other frameworks
---------------------------------------
DGL aims to be agnostic to existing-framework. It provides a narrow API that
can be ported on top of other tensor-based, autograd-enabled frameworks. Our
prototype works with MXNet, Gluon and Pytorch.

Free software
-------------
DGL is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/jermainewang/dgl>`_.

History
-------
Prototype of DGL started in early Spring, 2018, at NYU Shanghai by Prof. Zheng
Zhang and Quan Gan. Serious development began when Minjie, Lingfan and Prof
Jinyang Li from NYU's system group joined, flanked by a team of student
volunteers at NYU Shanghai, Fudan and other universities (Yu, Zihao, Murphy,
Allen, Qipeng, Qi, Hao), as well as early adopters at the CILVR lab (Jake
Zhao). Development accelerated when AWS MXNet Science team joined force, with
Da Zheng, Alex Smola, Haibin, Chao and a number of others. For full credit, see
[here](https://www.dgl.ai/ack)

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :glob:

   install/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :glob:

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :glob:

   api/python/index

Index
-----
* :ref:`genindex`
