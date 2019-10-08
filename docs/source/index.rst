.. DGL documentation master file, created by
   sphinx-quickstart on Fri Oct  5 14:18:01 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview of DGL
===============

Deep Graph Library (DGL) is a Python package built for easy implementation of
graph neural network model family, on top of existing DL frameworks (e.g.
PyTorch, MXNet, Gluon etc.).

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

Get Started
-----------

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :hidden:
   :glob:

   install/index

Follow the :doc:`instructions<install/index>` to install DGL. The :doc:`DGL at a glance<tutorials/basics/1_first>`
is the most common place to get started with. Each tutorial is accompanied with a runnable
python script and jupyter notebook that can be downloaded.

.. ================================================================================================
   (start) MANUALLY INCLUDE THE GENERATED TUTORIALS/BASIC/INDEX.RST HERE TO EMBED THE EXAMPLES
   ================================================================================================

.. raw:: html

    <div class="sphx-glr-thumbcontainer">

.. only:: html

    .. figure:: /tutorials/basics/images/thumb/sphx_glr_1_first_thumb.png

        :ref:`sphx_glr_tutorials_basics_1_first.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/basics/1_first

.. raw:: html

    <div class="sphx-glr-thumbcontainer">

.. only:: html

    .. figure:: /tutorials/basics/images/thumb/sphx_glr_2_basics_thumb.png

        :ref:`sphx_glr_tutorials_basics_2_basics.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/basics/2_basics

.. raw:: html

    <div class="sphx-glr-thumbcontainer">

.. only:: html

    .. figure:: /tutorials/basics/images/thumb/sphx_glr_3_pagerank_thumb.png

        :ref:`sphx_glr_tutorials_basics_3_pagerank.py`

.. raw:: html

    </div>

.. toctree::
   :hidden:

   /tutorials/basics/3_pagerank

.. raw:: html

    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer">

.. only:: html

    .. figure:: /tutorials/basics/images/thumb/sphx_glr_4_batch_thumb.png

        :ref:`sphx_glr_tutorials_basics_4_batch.py`

.. raw:: html

    </div>

.. toctree::
   :hidden:

   /tutorials/basics/4_batch

.. raw:: html

    <div class="sphx-glr-thumbcontainer">

.. only:: html

    .. figure:: /tutorials/hetero/images/thumb/sphx_glr_1_basics_thumb.png

        :ref:`sphx_glr_tutorials_hetero_1_basics.py`

.. raw:: html

    </div>

.. toctree::
   :hidden:

   /tutorials/hetero/1_basics

.. raw:: html

    <div style='clear:both'></div>


.. ================================================================================================
   (end) MANUALLY INCLUDE THE GENERATED TUTORIALS/BASIC/INDEX.RST HERE TO EMBED THE EXAMPLES
   ================================================================================================

Learning DGL through examples
-----------------------------

The model tutorials are categorized based on the way they utilize DGL APIs.

* :ref:`Graph Neural Network and its variant <tutorials1-index>`: Learn how to use DGL to train
  popular **GNN models** on one input graph.
* :ref:`Dealing with many small graphs <tutorials2-index>`: Learn how to **batch** many
  graph samples for max efficiency.
* :ref:`Generative models <tutorials3-index>`: Learn how to deal with **dynamically-changing graphs**.
* :ref:`Old (new) wines in new bottle <tutorials4-index>`: Learn how to combine DGL with tensor-based
  DGL framework in a flexible way. Explore new perspective on traditional models by graphs.
* :ref:`Training on giant graphs <tutorials5-index>`: Learn how to train graph neural networks
  on giant graphs.

Or go through all of them :doc:`here <tutorials/models/index>`.

.. toctree::
   :maxdepth: 1
   :caption: Features
   :hidden:
   :glob:

   features/builtin
   features/nn

.. toctree::
   :maxdepth: 3
   :caption: Model Tutorials
   :hidden:
   :glob:

   tutorials/models/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :glob:

   api/python/index

.. toctree::
   :maxdepth: 1
   :caption: Developer Notes
   :hidden:
   :glob:

   contribute
   developer/ffi

.. toctree::
   :maxdepth: 1
   :caption: Misc
   :hidden:
   :glob:

   faq
   env_var
   resources

Free software
-------------
DGL is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/dmlc/dgl>`_ and check out our
:doc:`contribution guidelines <contribute>`.

History
-------
Prototype of DGL started in early Spring, 2018, at NYU Shanghai by Prof. `Zheng
Zhang <https://shanghai.nyu.edu/academics/faculty/directory/zheng-zhang>`_ and
Quan Gan. Serious development began when `Minjie
<https://jermainewang.github.io/>`_, `Lingfan <https://cs.nyu.edu/~lingfan/>`_
and Prof. `Jinyang Li <http://www.news.cs.nyu.edu/~jinyang/>`_ from NYU's
system group joined, flanked by a team of student volunteers at NYU Shanghai,
Fudan and other universities (Yu, Zihao, Murphy, Allen, Qipeng, Qi, Hao), as
well as early adopters at the CILVR lab (Jake Zhao). Development accelerated
when AWS MXNet Science team joined force, with Da Zheng, Alex Smola, Haibin
Lin, Chao Ma and a number of others. For full credit, see `here
<https://www.dgl.ai/ack>`_.

Index
-----
* :ref:`genindex`
