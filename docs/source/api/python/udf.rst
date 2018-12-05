.. _apiudf:

User-defined function related data structures
==================================================

.. currentmodule:: dgl.udf
.. automodule:: dgl.udf

There are two types of user-defined functions in DGL:

* **Node UDF** of signature ``NodeBatch -> dict``. The argument represents
  a batch of nodes. The returned dictionary should have ``str`` type key and ``tensor``
  type values.
* **Edge UDF** of signature ``EdgeBatch -> dict``. The argument represents
  a batch of edges. The returned dictionary should have ``str`` type key and ``tensor``
  type values.

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
