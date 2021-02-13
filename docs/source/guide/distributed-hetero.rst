.. _guide-distributed-hetero:

7.3 Distributed Heterogeneous graph training
--------------------------------------------

DGL v0.6.0 provides an experimental support for distributed training on heterogeneous graphs.
In this version, each node or edge is assigned with two IDs: homogeneous ID and type-wise ID.
In DGL, a node or edge in a heterogeneous graph has a unique ID in its own node type or edge type.
We identify a node or edge with a tuple: node/edge type and type-wise ID. In distributed training,
a node or edge can be identified by a homogeneous ID, in addition to the tuple of node/edge type
and type-wise ID. The homogeneous ID is unique regardless of the node type and edge type.
DGL arranges nodes and edges so that all nodes of the same type have contiguous
homogeneous IDs.

Below is an example adjancency matrix of a heterogeneous graph showing the homogeneous ID assignment.
Here, the graph has two types of nodes (`T0` and `T1` ), and four types of edges (`R0`, `R1`, `R2`, `R3` ).
There are a total of 400 nodes in the graph and each type has 200 nodes. Nodes
of `T0` have IDs in [0,200), while nodes of `T1` have IDs in [200, 400).
In this example, if we use a tuple to identify the nodes, nodes of `T0` are identified as
(T0, type-wise ID), where type-wise ID falls in [0, 200); nodes of `T1` are identified as
(T1, type-wise ID), where type-wise ID also falls in [0, 200).

    ```
        0 <- T0 -> 200 <- T1 -> 400
     0  +-----------+------------+
        |           |            |
     ^  |    R0     |     R1     |
     T0 |           |            |
     v  |           |            |
    200 +-----------+------------+
        |           |            |
     ^  |    R2     |     R3     |
     T1 |           |            |
     v  |           |            |
    400 +-----------+------------+
    ```

7.3.1 Access distributed graph data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For distributed training, :class:`~dgl.distributed.DistGraph` supports the heterogeneous graph API
in :class:`~dgl.DGLGraph`. Below shows an example of getting node data of `T0` on some nodes
by using type-wise node IDs. When accessing data in :class:`~dgl.distributed.DistGraph`, a user
needs to use type-wise IDs and corresponding node types or edge types.

.. code:: python

    import dgl
    g = dgl.distributed.DistGraph('graph_name', part_config='data/graph_name.json')
    feat = g.nodes['T0'].data['feat'][type_wise_ids]

A user can create distributed tensors and distributed embeddings for a particular node type or
edge type. Distributed tensors and embeddings are split and stored in multiple machines. To create
one, a user needs to specify how it is partitioned with :class:`~dgl.distributed.PartitionPolicy`.
Below shows an example of creating a distributed tensor for node type `T0` and store it as node data
of node type `T0`.

.. code:: python

    g.nodes['T0'].data['feat1'] = dgl.distributed.DistTensor((g.number_of_nodes('T0'), 1), th.float32, 'feat1',
                                                             part_policy=g.get_node_partition_policy('T0'))

The partition policies used for creating distributed tensors and embeddings are initialized when a heterogeneous
graph is loaded into the graph server. A user cannot create a new partition policy at runtime. Therefore, a user
can only create distributed tensors or embeddings for a node type or edge type.
Accessing distributed tensors and embeddings also requires type-wise IDs.

7.3.2 Distributed sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^

DGL v0.6 uses homogeneous IDs in distributed sampling. **Note**: this will change in the future release.
DGL provides four APIs to convert node IDs and edge IDs between the homogeneous IDs and type-wise IDs: 

* :func:`~dgl.distributed.GraphPartitionBook.map_to_per_ntype`: convert a homogeneous node ID to type-wise ID and node type ID.
* :func:`~dgl.distributed.GraphPartitionBook.map_to_per_etype`: convert a homogeneous edge ID to type-wise ID and edge type ID.
* :func:`~dgl.distributed.GraphPartitionBook.map_to_homo_nid`: convert type-wise ID and node type to a homogeneous node ID.
* :func:`~dgl.distributed.GraphPartitionBook.map_to_homo_eid`: convert type-wise ID and edge type to a homogeneous edge ID.

Below shows an example of sampling a subgraph with :func:`~dgl.distributed.sample_neighbors` from a heterogeneous graph
with a node type called `paper`. We first convert type-wise node IDs to homogeneous node IDs. After sampling a subgraph
from the seed nodes, we convert homogeneous node IDs and edge IDs to type-wise IDs and also store type IDs as node data
or edge data.

.. code:: python

        gpb = g.get_partition_book()
        # We need to map the type-wise node IDs to homogeneous IDs.
        cur = gpb.map_to_homo_nid(seeds, 'paper')
        # For a heterogeneous input graph, the returned frontier is stored in
        # the homogeneous graph format.
        frontier = dgl.distributed.sample_neighbors(g, cur, fanout, replace=False)
        block = dgl.to_block(frontier, cur)
        cur = block.srcdata[dgl.NID]

        block.edata[dgl.EID] = frontier.edata[dgl.EID]
        # Map the homogeneous edge Ids to their edge type.
        block.edata[dgl.ETYPE], block.edata[dgl.EID] = gpb.map_to_per_etype(block.edata[dgl.EID])
        # Map the homogeneous node Ids to their node types and per-type Ids.
        block.srcdata[dgl.NTYPE], block.srcdata[dgl.NID] = gpb.map_to_per_ntype(block.srcdata[dgl.NID])
        block.dstdata[dgl.NTYPE], block.dstdata[dgl.NID] = gpb.map_to_per_ntype(block.dstdata[dgl.NID])

From node/edge type IDs, we can retrieve node/edge types. For example, `g.ntypes[node_type_id]`. Once we have
node/edge type and type-wise IDs, we can retrieve node/edge data from `DistGraph` for mini-batch computation.
