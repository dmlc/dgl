.. _apibatch_heterograph:

BatchedDGLHeteroGraph -- Enable batched graph operations for heterographs
=========================================================================

.. currentmodule:: dgl
.. autoclass:: BatchedDGLHeteroGraph

Merge and decompose
-------------------

.. autosummary::
    :toctree: ../../generated/

    batch_hetero
    unbatch_hetero

Query batch summary
----------------------

.. autosummary::
    :toctree: ../../generated/

    BatchedDGLHeteroGraph.batch_size
    BatchedDGLHeteroGraph.batch_num_nodes
    BatchedDGLHeteroGraph.batch_num_edges
