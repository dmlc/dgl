.. _api-distributed:

dgl.distributed
=================================

.. automodule:: dgl.distributed

DGL Distributed
---------------

Initialization and Exit
```````````````````````

.. autosummary::
    :toctree: ../../generated/

    initialize
    exit_client

Distributed Graph
-----------------

.. autoclass:: DistGraph

.. autosummary::
    :toctree: ../../generated/

    node_split
    edge_split

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

.. autosummary::
    :toctree: ../../generated/

    load_partition
    load_partition_book
    partition_graph

Distributed Sampling
--------------------

Distributed DataLoader
``````````````````````

.. currentmodule:: dgl.distributed.dist_dataloader

.. autoclass:: DistDataLoader

Distributed Neighbor Sampling
`````````````````````````````

.. currentmodule:: dgl.distributed.graph_services

.. autosummary::
    :toctree: ../../generated/

    sample_neighbors
    find_edges
    in_subgraph

