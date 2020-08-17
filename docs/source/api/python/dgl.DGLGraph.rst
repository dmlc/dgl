.. _apigraph:

dgl.DGLGraph
=====================================================

.. currentmodule:: dgl
.. class:: DGLGraph

Querying metagraph structure
----------------------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.ntypes
    DGLGraph.etypes
    DGLGraph.srctypes
    DGLGraph.dsttypes
    DGLGraph.canonical_etypes
    DGLGraph.metagraph
    DGLGraph.to_canonical_etype
    DGLGraph.get_ntype_id
    DGLGraph.get_etype_id

Querying graph structure
------------------------

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

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.formats
    DGLGraph.create_format_

Querying and manipulating index data type
-----------------------------------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.idtype
    DGLGraph.long
    DGLGraph.int

Using Node/edge features
------------------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.nodes
    DGLGraph.ndata
    DGLGraph.edges
    DGLGraph.edata
    DGLGraph.node_attr_schemes
    DGLGraph.edge_attr_schemes
    DGLGraph.set_n_initializer
    DGLGraph.set_e_initializer
    DGLGraph.srcnodes
    DGLGraph.dstnodes
    DGLGraph.srcdata
    DGLGraph.dstdata
    DGLGraph.local_var
    DGLGraph.local_scope

Transforming graph
------------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.subgraph
    DGLGraph.edge_subgraph
    DGLGraph.node_type_subgraph
    DGLGraph.edge_type_subgraph
    DGLGraph.__getitem__

Converting to other formats
---------------------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.adj
    DGLGraph.adjacency_matrix
    DGLGraph.inc
    DGLGraph.incidence_matrix

Computing with DGLGraph
-----------------------------

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

Querying batch summary
----------------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.batch_size
    DGLGraph.batch_num_nodes
    DGLGraph.batch_num_edges

Mutating topology
-----------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.add_nodes
    DGLGraph.add_edges
    DGLGraph.remove_nodes
    DGLGraph.remove_edges

Device Control
--------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.to
    DGLGraph.device
