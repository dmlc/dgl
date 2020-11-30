.. _guide-distributed-preprocessing:

7.1 Preprocessing for Distributed Training
------------------------------------------

DGL requires preprocessing the graph data for distributed training, including two steps:
1) partition a graph into subgraphs, 2) assign nodes/edges with new Ids. DGL provides
a partitioning API that performs the two steps. The API supports both random partitioning
and a `Metis <http://glaros.dtc.umn.edu/gkhome/views/metis>`__-based partitioning.
The benefit of Metis partitioning is that it can generate
partitions with minimal edge cuts that reduces network communication for distributed training
and inference. DGL uses the latest version of Metis with the options optimized for the real-world
graphs with power-law distribution. After partitioning, the API constructs the partitioned results
in a format that is easy to load during the training.

**Note**: The graph partition API currently runs on one machine. Therefore, if a graph is large,
users will need a large machine to partition a graph. In the future, DGL will support distributed
graph partitioning.

By default, the partition API assigns new IDs to the nodes and edges in the input graph to help locate
nodes/edges during distributed training/inference. After assigning IDs, the partition API shuffles
all node data and edge data accordingly. During the training, users just use the new node/edge IDs.
However, the original IDs are still accessible through ``g.ndata['orig_id']`` and ``g.edata['orig_id']``,
where ``g`` is a DistGraph object (see the section of DistGraph).

The partitioned results are stored in multiple files in the output directory. It always contains
a JSON file called xxx.json, where xxx is the graph name provided to the partition API. The JSON file
contains all the partition configurations. If the partition API does not assign new IDs to nodes and edges,
it generates two additional Numpy files: `node_map.npy` and `edge_map.npy`, which stores the mapping between
node/edge IDs and partition IDs. The Numpy arrays in the two files are large for a graph with billions of
nodes and edges because they have an entry for each node and edge in the graph. Inside the folders for
each partition, there are three files that store the partition data in the DGL format. `graph.dgl` stores
the graph structure of the partition as well as some metadata on nodes and edges. `node_feats.dgl` and
`edge_feats.dgl` stores all features of nodes and edges that belong to the partition. 

.. code-block:: none

    data_root_dir/
        |-- xxx.json                  # partition configuration file in JSON
        |-- node_map.npy       # partition id of each node stored in a numpy array (optional)
        |-- edge_map.npy       # partition id of each edge stored in a numpy array (optional)
        |-- part0/                     # data for partition 0
            |-- node_feats.dgl   # node features stored in binary format
            |-- edge_feats.dgl   # edge features stored in binary format
            |-- graph.dgl            # graph structure of this partition stored in binary format
        |-- part1/                      # data for partition 1
            |-- node_feats.dgl
            |-- edge_feats.dgl
            |-- graph.dgl

Load balancing
~~~~~~~~~~~~~~

When partitioning a graph, by default, Metis only balances the number of nodes in each partition.
This can result in suboptimal configuration, depending on the task at hand. For example, in the case
of semi-supervised node classification, a trainer performs computation on a subset of labeled nodes in
a local partition. A partitioning that only balances nodes in a graph (both labeled and unlabeled), may
end up with computational load imbalance. To get a balanced workload in each partition, the partition API
allows balancing between partitions with respect to the number of nodes in each node type, by specifying
``balance_ntypes`` in :func:`dgl.distributed.partition_graph`. Users can take advantage of this and consider
nodes in the training set, validation set and test set are of different node types.

The following example considers nodes inside the training set and outside the training set are two types of nodes:

.. code:: python

    dgl.distributed.partition_graph(g, ‘graph_name’, 4, ‘/tmp/test’, balance_ntypes=g.ndata[‘train_mask’])

In addition to balancing the node types, :func:`dgl.distributed.partition_graph` also allows balancing
between in-degrees of nodes of different node types by specifying ``balance_edges``. This balances
the number of edges incident to the nodes of different types.

**Note**: The graph name passed to :func:`dgl.distributed.partition_graph` is an important argument.
The graph name will be used by :class:`dgl.distributed.DistGraph` to identify a distributed graph.
A legal graph name should only contain alphabetic characters and underscores.
