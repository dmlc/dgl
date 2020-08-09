.. _apiudf:

dgl.udf
==================================================

.. currentmodule:: dgl.udf

User-defined functions (UDFs) are flexible ways to configure message passing computation.
There are two types of UDFs in DGL:

* **Node UDF** of signature ``NodeBatch -> dict``. The argument represents
  a batch of nodes. The returned dictionary should have ``str`` type key and ``tensor``
  type values.
* **Edge UDF** of signature ``EdgeBatch -> dict``. The argument represents
  a batch of edges. The returned dictionary should have ``str`` type key and ``tensor``
  type values.

The size of the batch dimension is determined by the DGL framework
for good efficiency and small memory footprint. Users should not make
assumption in the batch dimension.

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
    EdgeBatch.__len__

NodeBatch
---------

The class that can represent a batch of nodes.

.. autosummary::
    :toctree: ../../generated/

    NodeBatch.data
    NodeBatch.mailbox
    NodeBatch.nodes
    NodeBatch.batch_size
    NodeBatch.__len__
