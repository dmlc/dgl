.. _api-distributed:

dgl.distributed
=================================

.. automodule:: dgl.distributed

DGL Distributed
---------------

Initialization and Exit
```````````````````````

.. currentmodule:: dgl.distributed.dist_context

.. autoclass:: initialize
.. autoclass:: exit_client

Distributed Graph
-----------------

.. currentmodule:: dgl.distributed.dist_graph

.. autoclass:: DistGraph

.. autoclass:: node_split

.. autoclass:: edge_split

Partition
---------

Graph partition book
````````````````````

.. currentmodule:: dgl.distributed.graph_partition_book

.. autoclass:: GraphPartitionBook

.. autoclass:: PartitionPolicy

Split and Load Graphs
`````````````````````

.. currentmodule:: dgl.distributed.partition

.. autoclass:: load_partition

.. autoclass:: load_partition_book

.. autoclass:: partition_graph

Distributed Sampling
--------------------

Distributed DataLoader
``````````````````````

.. currentmodule:: dgl.distributed.dist_dataloader

.. autoclass:: DistDataLoader

Distributed Neighbor Sampling
`````````````````````````````

.. currentmodule:: dgl.distributed.graph_services

.. autoclass:: sample_neighbors

.. autoclass:: find_edges

.. autoclass:: in_subgraph

