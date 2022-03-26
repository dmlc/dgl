.. _api-distributed:

dgl.distributed
=================================

.. automodule:: dgl.distributed

Initialization
---------------

.. autosummary::
    :toctree: ../../generated/

    initialize

Distributed Graph
-----------------

.. autoclass:: DistGraph
    :members: ndata, edata, idtype, device, ntypes, etypes, number_of_nodes, number_of_edges, node_attr_schemes, edge_attr_schemes, rank, find_edges, get_partition_book, barrier, local_partition, num_nodes, num_edges, get_node_partition_policy, get_edge_partition_policy, get_etype_id, get_ntype_id, nodes, edges, out_degrees, in_degrees

Distributed Tensor
------------------

.. autoclass:: DistTensor
    :members: part_policy, shape, dtype, name

Distributed Node Embedding
---------------------
.. currentmodule:: dgl.distributed

.. autoclass:: DistEmbedding


Distributed embedding optimizer
-------------------------
.. currentmodule:: dgl.distributed.optim.pytorch

.. autoclass:: SparseAdagrad
    :members: step

.. autoclass:: SparseAdam
    :members: step

Distributed workload split
--------------------------

.. currentmodule:: dgl.distributed.dist_graph

.. autosummary::
    :toctree: ../../generated/

    node_split
    edge_split

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

Partition
---------

Graph partition book
````````````````````

.. currentmodule:: dgl.distributed.graph_partition_book

.. autoclass:: GraphPartitionBook
    :members: shared_memory, num_partitions, metadata, nid2partid, eid2partid, partid2nids, partid2eids, nid2localnid, eid2localeid, partid, map_to_per_ntype, map_to_per_etype, map_to_homo_nid, map_to_homo_eid

.. autoclass:: PartitionPolicy
    :members: policy_str, part_id, partition_book, to_local, to_partid, get_part_size, get_size

Split and Load Graphs
`````````````````````

.. currentmodule:: dgl.distributed.partition

.. autosummary::
    :toctree: ../../generated/

    load_partition
    load_partition_book
    partition_graph

