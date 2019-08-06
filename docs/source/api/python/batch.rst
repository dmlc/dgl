.. _apibatch:

BatchedDGLGraph -- Enable batched graph operations
==================================================

.. currentmodule:: dgl
.. autoclass:: BatchedDGLGraph

Merge and decompose
-------------------

.. autosummary::
    :toctree: ../../generated/

    batch
    unbatch

Query batch summary
----------------------

.. autosummary::
    :toctree: ../../generated/

    BatchedDGLGraph.batch_size
    BatchedDGLGraph.batch_num_nodes
    BatchedDGLGraph.batch_num_edges

Graph Readout
-------------

.. autosummary::
    :toctree: ../../generated/

    sum_nodes
    sum_edges
    mean_nodes
    mean_edges
    max_nodes
    max_edges
    topk_nodes
    topk_edges
    softmax_nodes
    softmax_edges
    broadcast_nodes
    broadcast_edges
