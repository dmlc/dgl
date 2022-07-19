.. _guide-distributed-partition:

7.4 Graph Partitioning
---------------------------

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

Output Format
~~~~~~~~~~~~~~~~~~~~~~~~~~

Regardless of the partitioning algorithm in use, the partitioned results are stored
in data files organized as follows:

.. code-block:: none

    data_root_dir/
      |-- mygraph.json          # partition configuration file in JSON
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

.. warning::

    TBD: some tips for inspecting the data

Parallel METIS Partitioning
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
