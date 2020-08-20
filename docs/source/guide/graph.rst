.. _guide-graph:

Chapter 1: Graph
======================

Graphs express entities (nodes) along with their relations (edges), and both nodes and
edges can be typed (e.g., ``"user"`` and ``"item"`` are two different types of nodes). DGL provides a
graph-centric programming abstraction with its core data structure -- :class:`~dgl.DGLGraph`. :class:`~dgl.DGLGraph`
provides its interface to handle a graph's structure, its node/edge features, and the resulting
computations that can be performed using these components.

This chapter introduce some core concepts of :class:`~dgl.DGLGraph`:

- Graphs, Nodes, and Edges
- Feature Data
- Loading Graphs from Disk
- Heterogeneous Graphs
- Using :class:`~dgl.DGLGraph` on a GPU

Some Basic Definitions about Graphs (Graphs 101)
------------------------------------------------

A graph :math:`G=(V, E)` is a structure used to represent entities and their relations. It consists of
two sets -- the set of nodes :math:`V` (also called vertices) and the set of edges :math:`E` (also called
arcs). An edge :math:`(u, v) \in E` connecting a pair of nodes :math:`u` and :math:`v` indicates that there is a
relation between them. The relation can either be undirected, e.g., capturing symmetric
relations between nodes, or directed, capturing asymmetric relations. For example, if a
graph is used to model the friendships relations of people in a social network, then the edges
will be undirected as friendship is mutual; however, if the graph is used to model how people
follow each other on Twitter, then the edges are directed. Depending on the edges'
directionality, a graph can be *directed* or *undirected*.

Graphs can be *weighted* or *unweighted*. In a weighted graph, each edge is associated with a
scalar weight. For example, such weights might represent lengths or connectivity strengths.

Graphs can also be either *homogeneous* or *heterogeneous*. In a homogeneous graph, all
the nodes represent instances of the same type and all the edges represent relations of the
same type. For instance, a social network is a graph consisting of people and their
connections, representing the same entity type.

In contrast, in a heterogeneous graph, the nodes and edges can be of different types. For
instance, the graph encoding a marketplace will have buyer, seller, and product nodes that
are connected via wants-to-buy, has-bought, is-customer-of, and is-selling edges. The
bipartite graph is a special, commonly-used type of heterogeneous graph, where edges
exist between nodes of two different types. For example, in a recommender system, one can
use a bipartite graph to represent the interactions between users and items. For working
with heterogeneous graphs in DGL, see :ref:`hetero`.

Multigraphs are graphs that can have multiple (directed) edges between the same pair of nodes,
including self loops. For instance, two authors can coauthor a paper in different years,
resulting in edges with different features.

Graphs, Nodes, and Edges
------------------------

DGL represents each node by a unique integer, called its node ID, and each edge by a pair
of integers corresponding to the IDs of its end nodes. DGL assigns to each edge a unique
integer, called its **edge ID**, based on the order in which it was added to the graph. The
numbering of node and edge IDs starts from 0. In DGL, all the edges are directed, and an
edge :math:`(u, v)` indicates that the direction goes from node :math:`u` to node :math:`v`.

To specify multiple nodes, DGL uses a 1-D integer tensor (i.e., PyTorch's tensor,
TensorFlow's Tensor, or MXNet's ndarray) of node IDs. DGL calls this format "node-tensors".
To specify multiple edges, it uses a tuple of node-tensors :math:`(U, V)`. :math:`(U[i], V[i])`
decides an edge from :math:`U[i]` to :math:`V[i]`.

One way to create a :class:`~dgl.DGLGraph` is to use the :func:`dgl.graph` method, which takes
as input a set of edges. DGL also supports creating graphs from other data sources, see :ref:`external`.

The following code snippet uses the :func:`dgl.graph` method to create a :class:`~dgl.DGLGraph`
corresponding to the four-node graph shown below and illustrates some of its APIs for
querying the graph's structure.

.. figure:: https://data.dgl.ai/asset/image/user_guide_graphch_1.png
    :height: 200px
    :width: 300px
    :align: center

.. code::

    >>> import dgl
    >>> import torch as th

    >>> # edges 0->1, 0->2, 0->3, 1->3
    >>> u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
    >>> g = dgl.graph((u, v))
    >>> print(g) # number of nodes are inferred from the max node IDs in the given edges
    Graph(num_nodes=4, num_edges=4,
          ndata_schemes={}
          edata_schemes={})

    >>> # Node IDs
    >>> print(g.nodes())
    tensor([0, 1, 2, 3])
    >>> # Edge end nodes
    >>> print(g.edges())
    (tensor([0, 0, 0, 1]), tensor([1, 2, 3, 3]))
    >>> # Edge end nodes and edge IDs
    >>> print(g.edges(form='all'))
    (tensor([0, 0, 0, 1]), tensor([1, 2, 3, 3]), tensor([0, 1, 2, 3]))

    >>> # If the node with the largest ID is isolated (meaning no edges),
    >>> # then one needs to explicitly set the number of nodes
    >>> g = dgl.graph((u, v), num_nodes=8)

For an undirected graph, one needs to create edges for both directions. :func:`dgl.to_bidirected`
can be helpful in this case, which converts a graph into a new one with edges for both directions.

.. code::

    >>> bg = dgl.to_bidirected(g)
    >>> bg.edges()
    (tensor([0, 0, 0, 1, 1, 2, 3, 3]), tensor([1, 2, 3, 0, 3, 0, 0, 1]))

.. note::

    Tensor types are generally preferred throughout DGL APIs due to their efficient internal
    storage in C and explicit data type and device context information. However, most DGL APIs
    do support python iterable (e.g., list) or numpy.ndarray as arguments for quick prototyping.

DGL can use either :math:`32`- or :math:`64`-bit integers to store the node and edge IDs. The data types for
the node and edge IDs should be the same. By using :math:`64` bits, DGL can handle graphs with
up to :math:`2^{63} - 1` nodes or edges. However, if a graph contains less than :math:`2^{31} - 1` nodes or edges,
one should use :math:`32`-bit integers as it leads to better speed and requires less memory.
DGL provides methods for making such conversions. See below for an example.

.. code::

    >>> edges = th.tensor([2, 5, 3]), th.tensor([3, 5, 0])  # edges 2->3, 5->5, 3->0
    >>> g64 = dgl.graph(edges)  # DGL uses int64 by default
    >>> print(g64.idtype)
    torch.int64
    >>> g32 = dgl.graph(edges, idtype=th.int32)  # create a int32 graph
    >>> g32.idtype
    torch.int32
    >>> g64_2 = g32.long()  # convert to int64
    >>> g64_2.idtype
    torch.int64
    >>> g32_2 = g64.int()  # convert to int32
    >>> g32_2.idtype
    torch.int32

See APIs: :func:`dgl.graph`, :func:`dgl.DGLGraph.nodes`, :func:`dgl.DGLGraph.edges`, :func:`dgl.to_bidirected`,
:func:`dgl.DGLGraph.int`, :func:`dgl.DGLGraph.long`, and :py:attr:`dgl.DGLGraph.idtype`.

Node and Edge Features
----------------------

The nodes and edges of a :class:`~dgl.DGLGraph` can have several user-defined named features for
storing graph-specific properties of the nodes and edges. These features can be accessed
via the :py:attr:`~dgl.DGLGraph.ndata` and :py:attr:`~dgl.DGLGraph.edata` interface. For example, the following code creates two node
features (named ``'x'`` and ``'y'`` in lines 5 and 9) and one edge feature (named ``'x'`` in line 6).

.. code::

    01. >>> import dgl
    02. >>> import torch as th
    03. >>> g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0])) # 6 nodes, 4 edges
    04. >>> g
        Graph(num_nodes=6, num_edges=4,
              ndata_schemes={}
              edata_schemes={})
    05. >>> g.ndata['x'] = th.ones(g.num_nodes(), 3)               # node feature of length 3
    06. >>> g.edata['x'] = th.ones(g.num_edges(), dtype=th.int32)  # scalar integer feature
    07. >>> g
        Graph(num_nodes=6, num_edges=4,
              ndata_schemes={'x' : Scheme(shape=(3,), dtype=torch.float32)}
              edata_schemes={'x' : Scheme(shape=(,), dtype=torch.int32)})
    08. >>> # different names can have different shapes
    09. >>> g.ndata['y'] = th.randn(g.num_nodes(), 5)
    10. >>> g.ndata['x'][1]                  # get node 1's feature
        tensor([1., 1., 1.])
    11. >>> g.edata['x'][th.tensor([0, 3])]  # get features of edge 0 and 3
        tensor([1, 1], dtype=torch.int32)

Important facts about the :py:attr:`~dgl.DGLGraph.ndata`/:py:attr:`~dgl.DGLGraph.edata` interface:

- Only features of numerical types (e.g., float, double, and int) are allowed. They can
  be scalars, vectors or multi-dimensional tensors.
- Each node feature has a unique name and each edge feature has a unique name.
  The features of nodes and edges can have the same name. (e.g., 'x' in the above example).
- A feature is created via tensor assignment, which assigns a feature to each
  node/edge in the graph. The leading dimension of that tensor must be equal to the
  number of nodes/edges in the graph. You cannot assign a feature to a subset of the
  nodes/edges in the graph.
- Features of the same name must have the same dimensionality and data type.
- The feature tensor is in row-major layout -- each row-slice stores the feature of one
  node or edge (e.g., see lines 10-11 in the above example).

For weighted graphs, one can store the weights as an edge feature as below.

.. code::

    >>> # edges 0->1, 0->2, 0->3, 1->3
    >>> edges = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
    >>> weights = th.tensor([0.1, 0.6, 0.9, 0.7])  # weight of each edge
    >>> g = dgl.graph(edges)
    >>> g.edata['w'] = weights  # give it a name 'w'
    >>> g
    Graph(num_nodes=4, num_edges=4,
          ndata_schemes={}
          edata_schemes={'w' : Scheme(shape=(,), dtype=torch.float32)})

See APIs: :py:attr:`~dgl.DGLGraph.ndata`, :py:attr:`~dgl.DGLGraph.edata`.

.. _external:

Creating Graphs from External Sources
-------------------------------------

The options to construct a :class:`~dgl.DGLGraph` from external sources include:

- Conversion from external python libraries for graphs and sparse matrices (NetworkX and SciPy).
- Loading graphs from disk.

The section does not cover functions that generate graphs by transforming from other
graphs. See the API reference manual for an overview of them.

Creating Graphs from External Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following code snippet is an example for creating a graph from a SciPy sparse matrix and a NetworkX graph.

.. code::

    >>> import dgl
    >>> import torch as th
    >>> import scipy.sparse as sp
    >>> spmat = sp.rand(100, 100, density=0.05) # 5% nonzero entries
    >>> dgl.from_scipy(spmat)                   # from SciPy
    Graph(num_nodes=100, num_edges=500,
          ndata_schemes={}
          edata_schemes={})

    >>> import networkx as nx
    >>> nx_g = nx.path_graph(5) # a chain 0-1-2-3-4
    >>> dgl.from_networkx(nx_g) # from networkx
    Graph(num_nodes=5, num_edges=8,
          ndata_schemes={}
          edata_schemes={})

Note that when constructing from the `nx.path_graph(5)`, the resulting :class:`~dgl.DGLGraph` has 8
edges instead of 4. This is because `nx.path_graph(5)` constructs an undirected NetworkX graph
:class:`networkx.Graph` while a :class:`~dgl.DGLGraph` is always directed. In converting an undirected
NetworkX graph into a :class:`~dgl.DGLGraph`, DGL internally converts undirected edges to two directed edges.
Using directed NetworkX graphs :class:`networkx.DiGraph` can avoid such behavior.

.. code::

    >>> nxg = nx.DiGraph([(2, 1), (1, 2), (2, 3), (0, 0)])
    >>> dgl.from_networkx(nxg)
    Graph(num_nodes=4, num_edges=4,
          ndata_schemes={}
          edata_schemes={})

.. note::

    DGL internally converts SciPy matrices and NetworkX graphs to tensors to construct graphs.
    Hence, these construction methods are not meant for performance critical parts.

See APIs: :func:`dgl.from_scipy`, :func:`dgl.from_networkx`.

Loading Graphs from Disk
^^^^^^^^^^^^^^^^^^^^^^^^

There are many data formats for storing graphs and it isn't possible to enumerate every option.
Thus, this section only gives some general pointers on certain common ones.

Comma Separated Values (CSV)
""""""""""""""""""""""""""""

One very common format is CSV, which stores nodes, edges, and their features in a tabular format:

.. table:: nodes.csv

   +-----------+
   |age, title |
   +===========+
   |43, 1      |
   +-----------+
   |23, 3      |
   +-----------+
   |...        |
   +-----------+

.. table:: edges.csv

   +-----------------+
   |src, dst, weight |
   +=================+
   |0, 1, 0.4        |
   +-----------------+
   |0, 3, 0.9        |
   +-----------------+
   |...              |
   +-----------------+

There are known Python libraries (e.g. pandas) for loading this type of data into python
objects (e.g., :class:`numpy.ndarray`), which can then be used to construct a DGLGraph. If the
backend framework also provides utilities to save/load tensors from disk (e.g., :func:`torch.save`,
:func:`torch.load`), one can follow the same principle to build a graph.

See also: `Tutorial for loading a Karate Club Network from edge pairs CSV <https://github.com/dglai/WWW20-Hands-on-Tutorial/blob/master/basic_tasks/1_load_data.ipynb>`_.

JSON/GML Format
"""""""""""""""

Though not particularly fast, NetworkX provides many utilities to parse
`a variety of data formats <https://networkx.github.io/documentation/stable/reference/readwrite/index.html>`_
which indirectly allows DGL to create graphs from these sources.

DGL Binary Format
"""""""""""""""""

DGL provides APIs to save and load graphs from disk stored in binary format. Apart from the
graph structure, the APIs also handle feature data and graph-level label data. DGL also
supports checkpointing graphs directly to S3 or HDFS. The reference manual provides more
details about the usage.

See APIs: :func:`dgl.save_graphs`, :func:`dgl.load_graphs`.

.. _hetero:

Heterogeneous Graphs
--------------------

A heterogeneous graph can have nodes and edges of different types. Nodes/Edges of
different types have independent ID space and feature storage. For example in the figure below, the
user and game node IDs both start from zero and the they have different features.

.. figure:: https://data.dgl.ai/asset/image/user_guide_graphch_2.png

    An example heterogeneous graph with two types of nodes (user and game) and two types of edges (follows and plays).

Creating a Heterogeneous Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In DGL, a heterogeneous graph (heterograph for short) is specified with a series of graphs as below, one per
relation. Each relation is a string triplet ``(source node type, edge type, destination node type)``.
Since relations disambiguate the edge types, DGL calls them canonical edge types.

.. code::

    {relation1 : node_tensor_tuple1,
     relation2 : node_tensor_tuple2,
     ...}

The following code snippet is an example for creating a heterogeneous graph in DGL.

.. code::

    >>> import dgl
    >>> import torch as th

    >>> # Create a heterograph with 3 node types and 3 edges types.
    >>> graph_data = {
    ...    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
    ...    ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
    ...    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
    ... }
    >>> g = dgl.heterograph(graph_data)
    >>> g.ntypes
    ['disease', 'drug', 'gene']
    >>> g.etypes
    ['interacts', 'interacts', 'treats']
    >>> g.canonical_etypes
    [('drug', 'interacts', 'drug'),
     ('drug', 'interacts', 'gene'),
     ('drug', 'treats', 'disease')]

Note that homogeneous and bipartite graphs are just special heterogeneous graphs with one
relation.

.. code::

    >>> # A homogeneous graph
    >>> dgl.heterograph({('node_type', 'edge_type', 'node_type'): (u, v)})
    >>> # A bipartite graph
    >>> dgl.heterograph({('source_type', 'edge_type', 'destination_type'): (u, v)})

The *metagraph* associated with a heterogeneous graph is the schema of the graph. It specifies
type constraints on the sets of nodes and edges between the nodes. A node :math:`u` in a metagraph
corresponds to a node type in the associated heterograph. An edge :math:`(u, v)` in a metagraph indicates that
there are edges from nodes of type :math:`u` to nodes of type :math:`v` in the associated heterograph.

.. code::

    >>> g
    Graph(num_nodes={'disease': 3, 'drug': 3, 'gene': 4},
          num_edges={('drug', 'interacts', 'drug'): 2,
                     ('drug', 'interacts', 'gene'): 2,
                     ('drug', 'treats', 'disease'): 1},
          metagraph=[('drug', 'drug', 'interacts'),
                     ('drug', 'gene', 'interacts'),
                     ('drug', 'disease', 'treats')])
    >>> g.metagraph().edges()
    OutMultiEdgeDataView([('drug', 'drug'), ('drug', 'gene'), ('drug', 'disease')])

See APIs: :func:`dgl.heterograph`, :py:attr:`~dgl.DGLGraph.ntypes`, :py:attr:`~dgl.DGLGraph.etypes`,
:py:attr:`~dgl.DGLGraph.canonical_etypes`, :py:attr:`~dgl.DGLGraph.metagraph`.

Working with Multiple Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When multiple node/edge types are introduced, users need to specify the particular
node/edge type when invoking a DGLGraph API for type-specific information. In addition,
nodes/edges of different types have separate IDs.

.. code::

    >>> # Get the number of all nodes in the graph
    >>> g.num_nodes()
    10
    >>> # Get the number of drug nodes
    >>> g.num_nodes('drug')
    3
    >>> # Nodes of different types have separate IDs,
    >>> # hence not well-defined without a type specified
    >>> g.nodes()
    DGLError: Node type name must be specified if there are more than one node types.
    >>> g.nodes('drug')
    tensor([0, 1, 2])

To set/get features for a specific node/edge type, DGL provides two new types of syntax --
`g.nodes['node_type'].data['feat_name']` and `g.edges['edge_type'].data['feat_name']`.

.. code::

    >>> # Set/get feature 'hv' for nodes of type 'drug'
    >>> g.nodes['drug'].data['hv'] = th.ones(3, 1)
    >>> g.nodes['drug'].data['hv']
    tensor([[1.],
            [1.],
            [1.]])
    >>> # Set/get feature 'he' for edge of type 'treats'
    >>> g.edges['treats'].data['he'] = th.zeros(1, 1)
    >>> g.edges['treats'].data['he']
    tensor([[0.]])

If the graph only has one node/edge type, there is no need to specify the node/edge type.

.. code::

    >>> g = dgl.heterograph({
    ...    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
    ...    ('drug', 'is similar', 'drug'): (th.tensor([0, 1]), th.tensor([2, 3]))
    ... })
    >>> g.nodes()
    tensor([0, 1, 2, 3])
    >>> # To set/get feature with a single type, no need to use the new syntax
    >>> g.ndata['hv'] = th.ones(4, 1)

.. note::

    When the edge type uniquely determines the types of source and destination nodes, one
    can just use one string instead of a string triplet to specify the edge type. For example, for a
    heterograph with two relations ``('user', 'plays', 'game')`` and ``('user', 'likes', 'game')``, it
    is safe to just use ``'plays'`` or ``'likes'`` to refer to the two relations.

Loading Heterographs from Disk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Comma Separated Values (CSV)
""""""""""""""""""""""""""""

A common way to store a heterograph is to store nodes and edges of different types in different CSV files.
An example is as follows.

.. code::

    # data folder
    data/
    |-- drug.csv        # drug nodes
    |-- gene.csv        # gene nodes
    |-- disease.csv     # disease nodes
    |-- drug-interact-drug.csv  # drug-drug interaction edges
    |-- drug-interact-gene.csv  # drug-gene interaction edges
    |-- drug-treat-disease.csv  # drug-treat-disease edges

Similar to the case of homogeneous graphs, one can use packages like Pandas to parse
CSV files into numpy arrays or framework tensors, build a relation dictionary and
construct a heterograph from that. The approach also applies to other popular formats like
GML/JSON.

DGL Binary Format
"""""""""""""""""

DGL provides :func:`dgl.save_graphs` and :func:`dgl.load_graphs` respectively for saving
heterogeneous graphs in binary format and loading them from binary format.

Edge Type Subgraph
^^^^^^^^^^^^^^^^^^

One can create a subgraph of a heterogeneous graph by specifying the relations to retain, with
features copied if any.

.. code::

    >>> g = dgl.heterograph({
    ...    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
    ...    ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
    ...    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
    ... })
    >>> g.nodes['drug'].data['hv'] = th.ones(3, 1)

    >>> # Retain relations ('drug', 'interacts', 'drug') and ('drug', 'treats', 'disease')
    >>> # All nodes for 'drug' and 'disease' will be retained
    >>> eg = dgl.edge_type_subgraph(g, [('drug', 'interacts', 'drug'),
    ...                                 ('drug', 'treats', 'disease')])
    >>> eg
    Graph(num_nodes={'disease': 3, 'drug': 3},
          num_edges={('drug', 'interacts', 'drug'): 2, ('drug', 'treats', 'disease'): 1},
          metagraph=[('drug', 'drug', 'interacts'), ('drug', 'disease', 'treats')])
    >>> # The associated features will be copied as well
    >>> eg.nodes['drug'].data['hv']
    tensor([[1.],
            [1.],
            [1.]])

Converting Heterogeneous Graphs to Homogeneous Graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Heterographs provide a clean interface for managing nodes/edges of different types and
their associated features. This is particularly helpful when:

1. The features for nodes/edges of different types have different data types or sizes.
2. We want to apply different operations to nodes/edges of different types.

If the above conditions do not hold and one does not want to distinguish node/edge types in
modeling, then DGL allows converting a heterogeneous graph to a homogeneous graph with :func:`dgl.DGLGraph.to_homogeneous` API.
It proceeds as follows:

1. Relabels nodes/edges of all types using consecutive integers starting from 0
2. Merges the features across node/edge types specified by the user.

.. code::

    >>> g = dgl.heterograph({
    ...    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
    ...    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))})
    >>> g.nodes['drug'].data['hv'] = th.zeros(3, 1)
    >>> g.nodes['disease'].data['hv'] = th.ones(3, 1)
    >>> g.edges['interacts'].data['he'] = th.zeros(2, 1)
    >>> g.edges['treats'].data['he'] = th.zeros(1, 2)

    >>> # By default, it does not merge any features
    >>> hg = dgl.to_homogeneous(g)
    >>> 'hv' in hg.ndata
    False

    >>> # Copy edge features
    >>> # For feature copy, it expects features to have
    >>> # the same size and dtype across node/edge types
    >>> hg = dgl.to_homogeneous(g, edata=['he'])
    DGLError: Cannot concatenate column ‘he’ with shape Scheme(shape=(2,), dtype=torch.float32) and shape Scheme(shape=(1,), dtype=torch.float32)

    >>> # Copy node features
    >>> hg = dgl.to_homogeneous(g, ndata=['hv'])
    >>> hg.ndata['hv']
    tensor([[1.],
            [1.],
            [1.],
            [0.],
            [0.],
            [0.]])

    The original node/edge types and type-specific IDs are stored in :py:attr:`~dgl.DGLGraph.ndata` and :py:attr:`~dgl.DGLGraph.edata`.

.. code::

    >>> # Order of node types in the heterograph
    >>> g.ntypes
    ['disease', 'drug']
    >>> # Original node types
    >>> hg.ndata[dgl.NTYPE]
    tensor([0, 0, 0, 1, 1, 1])
    >>> # Original type-specific node IDs
    >>> hg.ndata[dgl.NID]
    >>> tensor([0, 1, 2, 0, 1, 2])

    >>> # Order of edge types in the heterograph
    >>> g.etypes
    ['interacts', 'treats']
    >>> # Original edge types
    >>> hg.edata[dgl.ETYPE]
    tensor([0, 0, 1])
    >>> # Original type-specific edge IDs
    >>> hg.edata[dgl.EID]
    tensor([0, 1, 0])

For modeling purposes, one may want to group some relations together and apply the same
operation to them. To address this need, one can first take an edge type subgraph of the
heterograph and then convert the subgraph to a homogeneous graph.

.. code::

    >>> g = dgl.heterograph({
    ...    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
    ...    ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
    ...    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
    ... })
    >>> sub_g = dgl.edge_type_subgraph(g, [('drug', 'interacts', 'drug'),
    ...                                    ('drug', 'interacts', 'gene')])
    >>> h_sub_g = dgl.to_homogeneous(sub_g)
    >>> h_sub_g
    Graph(num_nodes=7, num_edges=4,
          ...)

Using DGLGraph on a GPU
------------------------

One can create a :class:`~dgl.DGLGraph` on a GPU by passing two GPU tensors during construction.
Another approach is to use the :func:`~dgl.DGLGraph.to` API to copy a :class:`~dgl.DGLGraph` to a GPU, which
copies the graph structure as well as the feature data to the given device.

.. code::

    >>> import dgl
    >>> import torch as th
    >>> u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
    >>> g = dgl.graph((u, v))
    >>> g.ndata['x'] = th.randn(5, 3)  # original feature is on CPU
    >>> g.device
    device(type='cpu')
    >>> cuda_g = g.to('cuda:0')  # accepts any device objects from backend framework
    >>> cuda_g.device
    device(type='cuda', index=0)
    >>> cuda_g.ndata['x'].device       # feature data is copied to GPU too
    device(type='cuda', index=0)

    >>> # A graph constructed from GPU tensors is also on GPU
    >>> u, v = u.to('cuda:0'), v.to('cuda:0')
    >>> g = dgl.graph((u, v))
    >>> g.device
    device(type='cuda', index=0)

Any operations involving a GPU graph are performed on a GPU. Thus, they require all
tensor arguments to be placed on GPU already and the results (graph or tensor) will be on
GPU too. Furthermore, a GPU graph only accepts feature data on a GPU.

.. code::

    >>> cuda_g.in_degrees()
    tensor([0, 0, 1, 1, 1], device='cuda:0')
    >>> cuda_g.in_edges([2, 3, 4])   # ok for non-tensor type arguments
    (tensor([0, 1, 2], device='cuda:0'), tensor([2, 3, 4], device='cuda:0'))
    >>> cuda_g.in_edges(th.tensor([2, 3, 4]).to('cuda:0'))  # tensor type must be on GPU
    (tensor([0, 1, 2], device='cuda:0'), tensor([2, 3, 4], device='cuda:0'))
    >>> cuda_g.ndata['h'] = th.randn(5, 4)  # ERROR! feature must be on GPU too!
    DGLError: Cannot assign node feature "h" on device cpu to a graph on device
    cuda:0. Call DGLGraph.to() to copy the graph to the same device.
