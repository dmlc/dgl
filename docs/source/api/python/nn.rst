.. _apinn:

dgl.nn
==========

.. automodule:: dgl.nn

.. toctree::

   nn.pytorch
   nn.mxnet
   nn.tensorflow

dgl.nn.functional
=================

Edge Softmax module
-------------------

We also provide framework agnostic edge softmax module which was frequently used in
GNN-like structures, e.g. 
`Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`_,
`Transformer <https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>`_,
`Capsule <https://arxiv.org/pdf/1710.09829.pdf>`_, etc.

.. autosummary::
    :toctree: ../../generated/

   nn.functional.edge_softmax
