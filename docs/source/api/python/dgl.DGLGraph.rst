.. _apigraph:

dgl.DGLGraph
=====================================================

.. currentmodule:: dgl
.. class:: DGLGraph

    Class for storing graph structure and node/edge feature data.

    There are a few ways to create a DGLGraph:

    * To create a homogeneous graph from Tensor data, use :func:`dgl.graph`.
    * To create a heterogeneous graph from Tensor data, use :func:`dgl.heterograph`.
    * To create a graph from other data sources, use ``dgl.*`` create ops. See
      :ref:`api-graph-create-ops`.

    Read the user guide chapter :ref:`guide-graph` for an in-depth explanation about its
    usage.

Querying metagraph structure
----------------------------

Methods for getting information about the node and edge types. They are typically useful
when the graph is heterogeneous.

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.ntypes
    DGLGraph.etypes
    DGLGraph.srctypes
    DGLGraph.dsttypes
    DGLGraph.canonical_etypes
    DGLGraph.metagraph
    DGLGraph.to_canonical_etype

.. _apigraph-querying-graph-structure:

Querying graph structure
------------------------

Methods for getting information about the graph structure such as capacity, connectivity,
neighborhood, etc.

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.num_nodes
    DGLGraph.number_of_nodes
    DGLGraph.num_edges
    DGLGraph.number_of_edges
    DGLGraph.num_src_nodes
    DGLGraph.number_of_src_nodes
    DGLGraph.num_dst_nodes
    DGLGraph.number_of_dst_nodes
    DGLGraph.is_unibipartite
    DGLGraph.is_multigraph
    DGLGraph.is_homogeneous
    DGLGraph.has_nodes
    DGLGraph.has_edges_between
    DGLGraph.predecessors
    DGLGraph.successors
    DGLGraph.edge_ids
    DGLGraph.find_edges
    DGLGraph.in_edges
    DGLGraph.out_edges
    DGLGraph.in_degrees
    DGLGraph.out_degrees

Querying and manipulating sparse format
---------------------------------------

Methods for getting or manipulating the internal storage formats of a ``DGLGraph``.

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.formats
    DGLGraph.create_formats_

Querying and manipulating node/edge ID type
-----------------------------------------

Methods for getting or manipulating the data type for storing structure-related
data such as node and edge IDs.

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.idtype
    DGLGraph.long
    DGLGraph.int

Using Node/edge features
------------------------

Methods for getting or setting the data type for storing structure-related
data such as node and edge IDs.

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.nodes
    DGLGraph.ndata
    DGLGraph.edges
    DGLGraph.edata
    DGLGraph.node_attr_schemes
    DGLGraph.edge_attr_schemes
    DGLGraph.srcnodes
    DGLGraph.dstnodes
    DGLGraph.srcdata
    DGLGraph.dstdata

Transforming graph
------------------

Methods for generating a new graph by transforming the current ones. Most of them
are alias of the :ref:`api-subgraph-extraction` and :ref:`api-transform`
under the ``dgl`` namespace.

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.subgraph
    DGLGraph.edge_subgraph
    DGLGraph.node_type_subgraph
    DGLGraph.edge_type_subgraph
    DGLGraph.__getitem__
    DGLGraph.line_graph
    DGLGraph.reverse
    DGLGraph.add_self_loop
    DGLGraph.remove_self_loop
    DGLGraph.to_simple
    DGLGraph.to_cugraph
    DGLGraph.reorder_graph

Adjacency and incidence matrix
---------------------------------

Methods for getting the adjacency and the incidence matrix of the graph.

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.adj
    DGLGraph.adjacency_matrix
    DGLGraph.adj_tensors
    DGLGraph.adj_external
    DGLGraph.inc
    DGLGraph.incidence_matrix

Computing with DGLGraph
-----------------------------

Methods for performing message passing, applying functions on node/edge features, etc.

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.apply_nodes
    DGLGraph.apply_edges
    DGLGraph.send_and_recv
    DGLGraph.pull
    DGLGraph.push
    DGLGraph.update_all
    DGLGraph.multi_update_all
    DGLGraph.prop_nodes
    DGLGraph.prop_edges
    DGLGraph.filter_nodes
    DGLGraph.filter_edges

Querying and manipulating batch information
----------------------------------------------

Methods for getting/setting the batching information if the current graph is a batched
graph generated from :func:`dgl.batch`. They are also widely used in the
:ref:`api-batch`.

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.batch_size
    DGLGraph.batch_num_nodes
    DGLGraph.batch_num_edges
    DGLGraph.set_batch_num_nodes
    DGLGraph.set_batch_num_edges


Mutating topology
-----------------

Methods for mutating the graph structure *in-place*.

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.add_nodes
    DGLGraph.add_edges
    DGLGraph.remove_nodes
    DGLGraph.remove_edges

Device Control
--------------

Methods for getting or changing the device on which the graph is hosted.

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.to
    DGLGraph.device
    DGLGraph.cpu
    DGLGraph.pin_memory_
    DGLGraph.unpin_memory_
    DGLGraph.is_pinned

Misc
----

Other utility methods.

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.local_scope
