.. _guide-distributed-partition:

7.4 Advanced Graph Partitioning
---------------------------------------

The chapter covers some of the advanced topics for graph partitioning.

METIS partition algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`METIS <http://glaros.dtc.umn.edu/gkhome/views/metis>`__ is a state-of-the-art
graph partitioning algorithm that can generate partitions with minimal number
of cross-partition edges, making it suitable for distributed message passing
where the amount of network communication is proportional to the number of
cross-partition edges. DGL has integrated METIS as the default partitioning
algorithm in its :func:`dgl.distributed.partition_graph` API.

Load balancing
~~~~~~~~~~~~~~~~

When partitioning a graph, by default, METIS only balances the number of nodes
in each partition.  This can result in suboptimal configuration, depending on
the task at hand. For example, in the case of semi-supervised node
classification, a trainer performs computation on a subset of labeled nodes in
a local partition. A partitioning that only balances nodes in a graph (both
labeled and unlabeled), may end up with computational load imbalance. To get a
balanced workload in each partition, the partition API allows balancing between
partitions with respect to the number of nodes in each node type, by specifying
``balance_ntypes`` in :func:`~dgl.distributed.partition_graph`. Users can take
advantage of this and consider nodes in the training set, validation set and
test set are of different node types.

The following example considers nodes inside the training set and outside the
training set are two types of nodes:

.. code:: python

    dgl.distributed.partition_graph(g, 'graph_name', 4, '/tmp/test', balance_ntypes=g.ndata['train_mask'])

In addition to balancing the node types,
:func:`dgl.distributed.partition_graph` also allows balancing between
in-degrees of nodes of different node types by specifying ``balance_edges``.
This balances the number of edges incident to the nodes of different types.

ID mapping
~~~~~~~~~~~~~

After partitioning, :func:`~dgl.distributed.partition_graph` remap node
and edge IDs so that nodes of the same partition are aranged together
(in a consecutive ID range), making it easier to store partitioned node/edge
features. The API also automatically shuffles the node/edge features
according to the new IDs. However, some downstream tasks may want to
recover the original node/edge IDs (such as extracting the computed node
embeddings for later use). For such cases, pass ``return_mapping=True``
to :func:`~dgl.distributed.partition_graph`, which makes the API returns
the ID mappings between the remapped node/edge IDs and their origianl ones.
For a homogeneous graph, it returns two vectors. The first vector maps every new
node ID to its original ID; the second vector maps every new edge ID to
its original ID. For a heterogeneous graph, it returns two dictionaries of
vectors. The first dictionary contains the mapping for each node type; the
second dictionary contains the mapping for each edge type.

.. code:: python

    node_map, edge_map = dgl.distributed.partition_graph(g, 'graph_name', 4, '/tmp/test',
                                                         balance_ntypes=g.ndata['train_mask'],
                                                         return_mapping=True)
    # Let's assume that node_emb is saved from the distributed training.
    orig_node_emb = th.zeros(node_emb.shape, dtype=node_emb.dtype)
    orig_node_emb[node_map] = node_emb

Output format
~~~~~~~~~~~~~~~~~~~~~~~~~~

Regardless of the partitioning algorithm in use, the partitioned results are stored
in data files organized as follows:

.. code-block:: none

    data_root_dir/
      |-- graph_name.json       # partition configuration file in JSON
      |-- part0/                # data for partition 0
      |  |-- node_feats.dgl     # node features stored in binary format
      |  |-- edge_feats.dgl     # edge features stored in binary format
      |  |-- graph.dgl          # graph structure of this partition stored in binary format
      |
      |-- part1/                # data for partition 1
      |  |-- node_feats.dgl
      |  |-- edge_feats.dgl
      |  |-- graph.dgl
      |
      |-- ...                   # data for other partitions

When distributed to a cluster, the metadata JSON should be copied to all the machines
while the ``partX`` folders should be dispatched accordingly.

DGL provides a :func:`dgl.distributed.load_partition` function to load one partition
for inspection.

.. code:: python

  >>> import dgl
  >>> # load partition 0
  >>> part_data = dgl.distributed.load_partition('data_root_dir/graph_name.json', 0)
  >>> g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data  # unpack
  >>> print(g)
  Graph(num_nodes=966043, num_edges=34270118,
        ndata_schemes={'orig_id': Scheme(shape=(), dtype=torch.int64),
                       'part_id': Scheme(shape=(), dtype=torch.int64),
                       '_ID': Scheme(shape=(), dtype=torch.int64),
                       'inner_node': Scheme(shape=(), dtype=torch.int32)}
        edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64),
                       'inner_edge': Scheme(shape=(), dtype=torch.int8),
                       'orig_id': Scheme(shape=(), dtype=torch.int64)})

As mentioned in the `ID mapping`_ section, each partition carries auxiliary information
saved as ndata or edata such as original node/edge IDs, partition IDs, etc. Each partition
not only saves nodes/edges it owns, but also includes node/edges that are adjacent to
the partition (called **HALO** nodes/edges). The ``inner_node`` and ``inner_edge``
indicate whether a node/edge truely belongs to the partition (value is ``True``)
or is a HALO node/edge (value is ``False``).

The :func:`~dgl.distributed.load_partition` function loads all data at once. Users can
load features or the partition book using the :func:`dgl.distributed.load_partition_feats`
and :func:`dgl.distributed.load_partition_book` APIs respectively.


Parallel METIS partitioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For massive graphs where parallel preprocessing is desired, DGL supports
`ParMETIS <http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview>`__ as one
of the choices of partitioning algorithms.

ParMETIS Installation
^^^^^^^^^^^^^^^^^^^^^^
ParMETIS requires METIS and GKLib. Please follow the instructions `here
<https://github.com/KarypisLab/GKlib>`__ to compile and install GKLib. For
compiling and install METIS, please follow the instructions below to clone
METIS with GIT and compile it with int64 support.

.. code-block:: none

    git clone https://github.com/KarypisLab/METIS.git
    make config shared=1 cc=gcc prefix=~/local i64=1
    make install


For now, we need to compile and install ParMETIS manually. We clone the DGL branch of ParMETIS as follows:

.. code-block:: none

    git clone --branch dgl https://github.com/KarypisLab/ParMETIS.git

Then compile and install ParMETIS.

.. code-block:: none

    make config cc=mpicc prefix=~/local
    make install

Before running ParMETIS, we need to set two environment variables: `PATH` and `LD_LIBRARY_PATH`.

.. code-block:: none

    export PATH=$PATH:$HOME/local/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/local/lib/

.. warning::

    TBD: Shall we go ahead explain the input/output of ParMETIS?
