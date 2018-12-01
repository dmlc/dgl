.. _apiudf:

User-defined function related data structures
==================================================

.. currentmodule:: dgl.udf
.. automodule:: dgl.udf

EdgeBatch
---------

The class that can represent a batch of edges.

.. autosummary::
    :toctree: ../../generated/

    EdgeBatch.src
    EdgeBatch.dst
    EdgeBatch.data
    EdgeBatch.edges
    EdgeBatch.batch_size

NodeBatch
---------

The class that can represent a batch of nodes.

.. autosummary::
    :toctree: ../../generated/

    NodeBatch.data
    NodeBatch.mailbox
    NodeBatch.nodes
    NodeBatch.batch_size
