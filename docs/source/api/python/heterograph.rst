.. _apiheterograph:

DGLHeteroGraph -- Typed graph with node/edge features
=====================================================

.. currentmodule:: dgl
.. autoclass:: DGLHeteroGraph

Conversion to and from heterogeneous graphs
-----------------------------------------

.. automodule:: dgl.convert
.. currentmodule:: dgl

.. autosummary::
    :toctree: ../../generated/

    graph
    bipartite
    hetero_from_relations
    heterograph
    to_hetero
    to_homo
    to_networkx
    DGLHeteroGraph.adjacency_matrix
    DGLHeteroGraph.incidence_matrix

Querying metagraph structure
----------------------------

.. autosummary::
    :toctree: ../../generated/

    DGLHeteroGraph.ntypes
    DGLHeteroGraph.etypes
    DGLHeteroGraph.canonical_etypes
    DGLHeteroGraph.metagraph
    DGLHeteroGraph.to_canonical_etype
    DGLHeteroGraph.get_ntype_id
    DGLHeteroGraph.get_etype_id

Querying graph structure
------------------------

.. autosummary::
    :toctree: ../../generated/

    DGLHeteroGraph.number_of_nodes
    DGLHeteroGraph.number_of_edges
    DGLHeteroGraph.is_multigraph
    DGLHeteroGraph.is_readonly
    DGLHeteroGraph.has_node
    DGLHeteroGraph.has_nodes
    DGLHeteroGraph.has_edge_between
    DGLHeteroGraph.has_edges_between
    DGLHeteroGraph.predecessors
    DGLHeteroGraph.successors
    DGLHeteroGraph.edge_id
    DGLHeteroGraph.edge_ids
    DGLHeteroGraph.find_edges
    DGLHeteroGraph.in_edges
    DGLHeteroGraph.out_edges
    DGLHeteroGraph.all_edges
    DGLHeteroGraph.in_degree
    DGLHeteroGraph.in_degrees
    DGLHeteroGraph.out_degree
    DGLHeteroGraph.out_degrees

Using Node/edge features
------------------------

.. autosummary::
    :toctree: ../../generated/

    DGLHeteroGraph.nodes
    DGLHeteroGraph.ndata
    DGLHeteroGraph.edges
    DGLHeteroGraph.edata
    DGLHeteroGraph.node_attr_schemes
    DGLHeteroGraph.edge_attr_schemes
    DGLHeteroGraph.set_n_initializer
    DGLHeteroGraph.set_e_initializer
    DGLHeteroGraph.local_var
    DGLHeteroGraph.local_scope

Transforming graph
------------------

.. autosummary::
    :toctree: ../../generated/

    DGLHeteroGraph.subgraph
    DGLHeteroGraph.edge_subgraph
    DGLHeteroGraph.node_type_subgraph
    DGLHeteroGraph.edge_type_subgraph

Computing with DGLHeteroGraph
-----------------------------

.. autosummary::
    :toctree: ../../generated/

    DGLHeteroGraph.apply_nodes
    DGLHeteroGraph.apply_edges
    DGLHeteroGraph.group_apply_edges
    DGLHeteroGraph.send
    DGLHeteroGraph.recv
    DGLHeteroGraph.multi_recv
    DGLHeteroGraph.send_and_recv
    DGLHeteroGraph.multi_send_and_recv
    DGLHeteroGraph.pull
    DGLHeteroGraph.multi_pull
    DGLHeteroGraph.push
    DGLHeteroGraph.update_all
    DGLHeteroGraph.multi_update_all
    DGLHeteroGraph.prop_nodes
    DGLHeteroGraph.prop_edges
    DGLHeteroGraph.filter_nodes
    DGLHeteroGraph.filter_edges
    DGLHeteroGraph.to
