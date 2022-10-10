dgl.contrib.cugraph.nn
======================

.. automodule:: dgl.contrib.cugraph.nn

*cugraph-ops* is a low-level framework-agnostic library from NVIDIA that provides computational primitives for GNNs on GPUs.
This module offers accelerated models for various graph convolutional layers using the aggregation functions in **pylibcugraphops**, a Python wrapper of *cugraph-ops*.

These models are only suitable for DGL sampled graphs (message flow graphs) in mini-batch learning and **not** intended for full-graph use cases.
They feature the exact same interface as their counterparts in :ref:`apinn-pytorch`, except that an additional input ``fanout`` is needed.
Users should pass in the same ``fanout`` value used in neighbor sampling.
Finally, all models here require a PyTorch backend and only work on CUDA devices.

RelGraphConv
------------

.. autoclass:: RelGraphConv
    :members: forward
