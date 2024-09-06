.. _api-distributed:

dgl.distributed
=================================

.. currentmodule:: dgl.distributed

DGL distributed module contains classes and functions to support
distributed Graph Neural Network training and inference on a cluster of
machines.

This includes a few submodules:

* distributed data structures including distributed graph, distributed tensor
  and distributed embeddings.
* distributed sampling.
* distributed workload split at runtime.
* graph partition.


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

.. autoclass:: DistEmbedding


Distributed embedding optimizer
-------------------------

.. autoclass:: dgl.distributed.optim.SparseAdagrad
    :members: step, save, load

.. autoclass:: dgl.distributed.optim.SparseAdam
    :members: step, save, load

Distributed workload split
--------------------------

.. autosummary::
    :toctree: ../../generated/

    node_split
    edge_split

Distributed Sampling
--------------------

Distributed DataLoader
``````````````````````

.. autoclass:: NodeCollator

.. autoclass:: EdgeCollator

.. autoclass:: DistDataLoader

.. autoclass:: DistNodeDataLoader

.. autoclass:: DistEdgeDataLoader

.. _api-distributed-sampling-ops:
Distributed Graph Sampling Operators
```````````````````````````````````````

.. autosummary::
    :toctree: ../../generated/

    sample_neighbors
    sample_etype_neighbors
    find_edges
    in_subgraph

Partition
---------

Graph partition book
````````````````````

.. autoclass:: GraphPartitionBook
    :members: shared_memory, num_partitions, metadata, nid2partid, eid2partid, partid2nids, partid2eids, nid2localnid, eid2localeid, partid, map_to_per_ntype, map_to_per_etype, map_to_homo_nid, map_to_homo_eid, canonical_etypes

.. autoclass:: PartitionPolicy
    :members: policy_str, part_id, partition_book, to_local, to_partid, get_part_size, get_size

Split and Load Partitions
````````````````````````````

.. autosummary::
    :toctree: ../../generated/

    load_partition
    load_partition_feats
    load_partition_book
    partition_graph
    dgl_partition_to_graphbolt
