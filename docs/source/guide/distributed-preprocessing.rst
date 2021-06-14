.. _guide-distributed-preprocessing:

7.1 Preprocessing for Distributed Training
------------------------------------------

:ref:`(中文版) <guide_cn-distributed-preprocessing>`

DGL requires to preprocess the graph data for distributed training. This includes two steps:
1) partition a graph into subgraphs, 2) assign nodes/edges with new IDs. For relatively small
graphs, DGL provides a partitioning API :func:`dgl.distributed.partition_graph` that performs
the two steps above. The API runs on one machine. Therefore, if a graph is large, users will
need a large machine to partition a graph when using this API. In addition to this API, we also
provide a solution to partition a large graph in a cluster of machines below (see Section 7.1.1).

:func:`dgl.distributed.partition_graph` supports both random partitioning
and a `Metis <http://glaros.dtc.umn.edu/gkhome/views/metis>`__-based partitioning.
The benefit of Metis partitioning is that it can generate
partitions with minimal edge cuts to reduce network communication for distributed training
and inference. DGL uses the latest version of Metis with the options optimized for the real-world
graphs with power-law distribution. After partitioning, the API constructs the partitioned results
in a format that is easy to load during the training.

By default, the partition API assigns new IDs to the nodes and edges in the input graph to help locate
nodes/edges during distributed training/inference. After assigning IDs, the partition API shuffles
all node data and edge data accordingly. After generating partitioned subgraphs, each subgraph is stored
as a ``DGLGraph`` object. The original node/edge IDs before reshuffling are stored in the field of
'orig_id' in the node/edge data of the subgraphs. The node data `dgl.NID` and the edge data `dgl.EID`
of the subgraphs store new node/edge IDs of the full graph after nodes/edges reshuffle.
During the training, users just use the new node/edge IDs.

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
        |-- node_map.npy              # partition id of each node stored in a numpy array (optional)
        |-- edge_map.npy              # partition id of each edge stored in a numpy array (optional)
        |-- part0/                    # data for partition 0
            |-- node_feats.dgl        # node features stored in binary format
            |-- edge_feats.dgl        # edge features stored in binary format
            |-- graph.dgl             # graph structure of this partition stored in binary format
        |-- part1/
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

    dgl.distributed.partition_graph(g, 'graph_name', 4, '/tmp/test', balance_ntypes=g.ndata['train_mask'])

In addition to balancing the node types, :func:`dgl.distributed.partition_graph` also allows balancing
between in-degrees of nodes of different node types by specifying ``balance_edges``. This balances
the number of edges incident to the nodes of different types.

**Note**: The graph name passed to :func:`dgl.distributed.partition_graph` is an important argument.
The graph name will be used by :class:`dgl.distributed.DistGraph` to identify a distributed graph.
A legal graph name should only contain alphabetic characters and underscores.

ID mapping
~~~~~~~~~~

:func:`dgl.distributed.partition_graph` shuffles node IDs and edge IDs during the partitioning and shuffles
node data and edge data accordingly. After training, we may need to save the computed node embeddings for
any downstream tasks. Therefore, we need to reshuffle the saved node embeddings according to their original
IDs.

When `return_mapping=True`, :func:`dgl.distributed.partition_graph` returns the mappings between shuffled
node/edge IDs and their original IDs. For a homogeneous graph, it returns two vectors. The first
vector maps every shuffled node ID to its original ID; the second vector maps every shuffled edge ID to its
original ID. For a heterogeneous graph, it returns two dictionaries of vectors. The first dictionary contains
the mapping for each node type; the second dictionary contains the mapping for each edge type.

.. code:: python

    node_map, edge_map = dgl.distributed.partition_graph(g, 'graph_name', 4, '/tmp/test',
                                                         balance_ntypes=g.ndata['train_mask'],
                                                         return_mapping=True)
    # Let's assume that node_emb is saved from the distributed training.
    orig_node_emb = th.zeros(node_emb.shape, dtype=node_emb.dtype)
    orig_node_emb[node_map] = node_emb


7.1.1 Distributed partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a large graph, DGL uses `ParMetis <http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview>`__ to partition
a graph in a cluster of machines. This solution requires users to prepare data for ParMETIS and use a DGL script
`tools/convert_partition.py` to construct :class:`dgl.DGLGraph` for the partitions output by ParMETIS.

**Note**: `convert_partition.py` uses the `pyarrow` package to load csv files. Please install `pyarrow`.

ParMETIS Installation
~~~~~~~~~~~~~~~~~~~~~

ParMETIS requires METIS and GKLib. Please follow the instructions `here <https://github.com/KarypisLab/GKlib>`__
to compile and install GKLib. For compiling and install METIS, please follow the instructions below to
clone METIS with GIT and compile it with int64 support.

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

Input format for ParMETIS
~~~~~~~~~~~~~~~~~~~~~~~~~

The input graph for ParMETIS is stored in three files with the following names: `xxx_nodes.txt`,
`xxx_edges.txt` and `xxx_stats.txt`, where `xxx` is a graph name.

Each row in `xxx_nodes.txt` stores the information of a node with the following format:

.. code-block:: none

    <node_type> <weight1> ... <orig_type_node_id> <attributes>

All fields are separated by whitespace:

* `<node_type>` is an integer. For a homogeneous graph, its value is always 0. For heterogeneous graphs,
  its value indicates the type of each node.
* `<weight1>`, `<weight2>`, etc are integers that indicate the node weights used by ParMETIS to balance
  graph partitions. If a user does not provide node weights, ParMETIS partitions a graph and balance
  the number of nodes in each partition (it is important to balance graph partitions in order to achieve
  good training speed). However, this default strategy may not be sufficient for many use cases.
  For example, in a heterogeneous graph, we want to partition the graph so that all partitions have
  roughly the same number of nodes for each node type. The toy example below shows how we can use
  node weights to balance the number of nodes of different types.
* `<orig_type_node_id>` is an integer representing the node ID in its own type. In DGL, nodes of each type
  are assigned with IDs starting from 0. For a homogeneous graph, this field is the same as the node ID. 
* `<attributes>` are optional fields. They can be used to store any values and ParMETIS does not interpret
  these fields. Potentially, we can store the node features and edge features in these fields for
  homogeneous graphs.
* The row ID indicates the *homogeneous* ID of nodes in a graph (all nodes are assigned with a unique ID).
  All nodes of the same type should be assigned with contiguous IDs. That is, nodes of the same type should
  be stored together in `xxx_nodes.txt`.

Below shows an example of a node file for a heterogeneous graph with two node types. Node type 0 has three
nodes; node type 1 has four nodes. It uses two node weights to ensure that ParMETIS will generate partitions
with roughly the same number of nodes for type 0 and the same number of nodes for type 1.

.. code-block:: none

    0 1 0 0
    0 1 0 1
    0 1 0 2
    1 0 1 0
    1 0 1 1
    1 0 1 2
    1 0 1 3

Similarly, each row in `xxx_edges.txt` stores the information of an edge with the following format:

.. code-block:: none

    <src_id> <dst_id> <type_edge_id> <edge_type> <attributes>

All fields are separated by whitespace:

* `<src_id>` is the *homogeneous* ID of the source node.
* `<dst_id>` is the *homogeneous* ID of the destination node.
* `<type_edge_id>` is the edge ID for the edge type.
* `<edge_type>` is the edge type.
* `<attributes>` are optional fields. They can be used to store any values and ParMETIS does not
  interpret these fields.

**Note**: please make sure that there are no duplicated edges and self-loop edges in the edge file.

`xxx_stats.txt` stores some basic statistics of the graph. It has only one line with three fields
separated by whitespace:

.. code-block:: none

    <num_nodes> <num_edges> <num_node_weights>

* `num_nodes` stores the total number of nodes regardless of node types.
* `num_edges` stores the total number of edges regardless of edge types.
* `num_node_weights` stores the number of node weights in the node file.

Run ParMETIS and output formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ParMETIS contains a command called `pm_dglpart`, which loads the graph stored in the three
files from the machine where `pm_dglpart` is invoked, distributes data to all machines in
the cluster and invokes ParMETIS to partition the graph. When it completes, it generates
three files for each partition: `p<part_id>-xxx_nodes.txt`, `p<part_id>-xxx_edges.txt`,
`p<part_id>-xxx_stats.txt`.

**Note**: ParMETIS reassigns IDs to nodes during the partitioning. After ID reassignment,
the nodes in a partition are assigned with contiguous IDs; furthermore, the nodes of
the same type are assigned with contiguous IDs.

`p<part_id>-xxx_nodes.txt` stores the node data of the partition. Each row represents
a node with the following fields:

.. code-block:: none

    <node_id> <node_type> <weight1> ... <orig_type_node_id> <attributes>

* `<node_id>` is the *homogeneous* node IDs after ID reassignment.
* `<node_type>` is the node type.
* `<weight1>` is the node weight used by ParMETIS.
* `<orig_type_node_id>` is the original node ID for a specific node type in the input heterogeneous graph.
* `<attributes>` are optional fields that contain any node attributes in the input node file.

`p<part_id>-xxx_edges.txt` stores the edge data of the partition. Each row represents
an edge with the following fields:

.. code-block:: none

    <src_id> <dst_id> <orig_src_id> <orig_dst_id> <orig_type_edge_id> <edge_type> <attributes>

* `<src_id>` is the *homogeneous* ID of the source node after ID reassignment.
* `<dst_id>` is the *homogeneous* ID of the destination node after ID reassignment.
* `<orig_src_id>` is the *homogeneous* ID of the source node in the input graph.
* `<orig_dst_id>` is the *homogeneous* ID of the destination node in the input graph.
* `<orig_type_edge_id>` is the edge ID for the specific edge type in the input graph.
* `<edge_type>` is the edge type.
* `<attributes>` are optional fields that contain any edge attributes in the input edge file.

When invoking `pm_dglpart`, the three input files: `xxx_nodes.txt`, `xxx_edges.txt`, `xxx_stats.txt`
should be located in the directory where `pm_dglpart` runs. The following command run four ParMETIS
processes to partition the graph named `xxx` into eight partitions (each process handles two partitions).

.. code-block:: none

    mpirun -np 4 pm_dglpart xxx 2

Convert ParMETIS outputs to DGLGraph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DGL provides a script named `convert_partition.py`, located in the `tools` directory, to convert the data
in the partition files into :class:`dgl.DGLGraph` objects and save them into files.
**Note**: `convert_partition.py` runs in a single machine. In the future, we will extend it to convert
graph data in parallel across multiple machines. **Note**: please install the `pyarrow` package
for loading data in csv files.

`convert_partition.py` has the following arguments:

* `--input-dir INPUT_DIR` specifies the directory that contains the partition files generated by ParMETIS.
* `--graph-name GRAPH_NAME` specifies the graph name.
* `--schema SCHEMA` provides a file that specifies the schema of the input heterogeneous graph.
  The schema file is a JSON file that lists node types and edge types as well as homogeneous ID ranges
  for each node type and edge type.
* `--num-parts NUM_PARTS` specifies the number of partitions.
* `--num-node-weights NUM_NODE_WEIGHTS` specifies the number of node weights used by ParMETIS
  to balance partitions.
* `[--workspace WORKSPACE]` is an optional argument that specifies a workspace directory to
  store some intermediate results.
* `[--node-attr-dtype NODE_ATTR_DTYPE]` is an optional argument that specifies the data type of
  node attributes in the remaining fields `<attributes>` of the node files.
* `[--edge-attr-dtype EDGE_ATTR_DTYPE]` is an optional argument that specifies the data type of
  edge attributes in the remaining fields `<attributes>` of the edge files.
* `--output OUTPUT` specifies the output directory that stores the partition results.

`convert_partition.py` outputs files as below:

.. code-block:: none

    data_root_dir/
        |-- xxx.json                  # partition configuration file in JSON
        |-- part0/                    # data for partition 0
            |-- node_feats.dgl        # node features stored in binary format (optional)
            |-- edge_feats.dgl        # edge features stored in binary format (optional)
            |-- graph.dgl             # graph structure of this partition stored in binary format
        |-- part1/
            |-- node_feats.dgl
            |-- edge_feats.dgl
            |-- graph.dgl

**Note**: if the data type of node attributes or edge attributes is specified, `convert_partition.py`
assumes all nodes/edges of any types have exactly these attributes. Therefore, if
nodes or edges of different types contain different numbers of attributes, users need to construct
them manually.

Below shows an example of the schema of the OGBN-MAG graph for `convert_partition.py`. It has two fields:
"nid" and "eid". Inside "nid", it lists all node types and the homogeneous ID ranges for each node type;
inside "eid", it lists all edge types and the homogeneous ID ranges for each edge type.

.. code-block:: none

    {
    "nid": {
        "author": [
            0,
            1134649
        ],
        "field_of_study": [
            1134649,
            1194614
        ],
        "institution": [
            1194614,
            1203354
        ],
        "paper": [
            1203354,
            1939743
        ]
    },
    "eid": {
        "affiliated_with": [
            0,
            1043998
        ],
        "writes": [
            1043998,
            8189658
        ],
        "rev-has_topic": [
            8189658,
            15694736
        ],
        "rev-affiliated_with": [
            15694736,
            16738734
        ],
        "cites": [
            16738734,
            22155005
        ],
        "has_topic": [
            22155005,
            29660083
        ],
        "rev-cites": [
            29660083,
            35076354
        ],
        "rev-writes": [
            35076354,
            42222014
        ]
    }
    }

Below shows the demo code to construct the schema file.

.. code-block:: none

    nid_ranges = {}
    eid_ranges = {}
    for ntype in hg.ntypes:
        ntype_id = hg.get_ntype_id(ntype)
        nid = th.nonzero(g.ndata[dgl.NTYPE] == ntype_id, as_tuple=True)[0]
        nid_ranges[ntype] = [int(nid[0]), int(nid[-1] + 1)]

    for etype in hg.etypes:
        etype_id = hg.get_etype_id(etype)
        eid = th.nonzero(g.edata[dgl.ETYPE] == etype_id, as_tuple=True)[0]
        eid_ranges[etype] = [int(eid[0]), int(eid[-1] + 1)]
    with open('mag.json', 'w') as outfile:
        json.dump({'nid': nid_ranges, 'eid': eid_ranges}, outfile, indent=4)

Construct node/edge features for a heterogeneous graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`dgl.DGLGraph` output by `convert_partition.py` stores a heterogeneous graph partition
as a homogeneous graph. Its node data contains a field called `orig_id` to store the node IDs
of a specific node type in the original heterogeneous graph and a field of `NTYPE` to store
the node type. In addition, it contains node data called `inner_node` that indicates
whether a node in the graph partition is assigned to the partition. If a node is assigned
to the partition, `inner_node` has 1; otherwise, its value is 0. Note: a graph partition
also contains some HALO nodes, which are assigned to other partitions but are connected with
some edges in this graph partition. By using this information, we can construct node features
for each node type separately and store them in a dictionary whose keys are
`<node_type>/<feature_name>` and values are node feature tensors. The code below illustrates
the construction of node feature dictionary. After the dictionary of tensors are constructed,
they are saved into a file.

.. code-block:: none

    node_data = {}
    for ntype in hg.ntypes:
        local_node_idx = th.logical_and(part.ndata['inner_node'].bool(),
                                        part.ndata[dgl.NTYPE] == hg.get_ntype_id(ntype))
        local_nodes = part.ndata['orig_id'][local_node_idx].numpy()
        for name in hg.nodes[ntype].data:
            node_data[ntype + '/' + name] = hg.nodes[ntype].data[name][local_nodes]
    dgl.data.utils.save_tensors(metadata['part-{}'.format(part_id)]['node_feats'], node_data)

We can construct the edge features in a very similar way. The only difference is that
all edges in the :class:`dgl.DGLGraph` object belong to the partition. So the construction
is even simpler.

.. code-block:: none

    edge_data = {}
    for etype in hg.etypes:
        local_edges = subg.edata['orig_id'][subg.edata[dgl.ETYPE] == hg.get_etype_id(etype)]
        for name in hg.edges[etype].data:
            edge_data[etype + '/' + name] = hg.edges[etype].data[name][local_edges]
    dgl.data.utils.save_tensors(metadata['part-{}'.format(part_id)]['edge_feats'], edge_data)
