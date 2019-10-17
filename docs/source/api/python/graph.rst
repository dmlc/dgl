.. _apigraph:

DGLGraph -- Untyped graph with node/edge features
=========================================

.. currentmodule:: dgl
.. autoclass:: DGLGraph

Adding nodes and edges
----------------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.add_nodes
    DGLGraph.add_edge
    DGLGraph.add_edges
    DGLGraph.clear

Querying graph structure
------------------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.number_of_nodes
    DGLGraph.number_of_edges
    DGLGraph.__len__
    DGLGraph.is_multigraph
    DGLGraph.has_node
    DGLGraph.has_nodes
    DGLGraph.__contains__
    DGLGraph.has_edge_between
    DGLGraph.has_edges_between
    DGLGraph.predecessors
    DGLGraph.successors
    DGLGraph.edge_id
    DGLGraph.edge_ids
    DGLGraph.find_edges
    DGLGraph.in_edges
    DGLGraph.out_edges
    DGLGraph.all_edges
    DGLGraph.in_degree
    DGLGraph.in_degrees
    DGLGraph.out_degree
    DGLGraph.out_degrees

Removing nodes and edges
------------------------

.. autosummary::
    :toctree: ../../generated/
    
    DGLGraph.remove_nodes
    DGLGraph.remove_edges

Transforming graph
------------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.subgraph
    DGLGraph.subgraphs
    DGLGraph.edge_subgraph
    DGLGraph.line_graph
    DGLGraph.reverse
    DGLGraph.readonly

Converting from/to other format
-------------------------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.to_networkx
    DGLGraph.from_networkx
    DGLGraph.from_scipy_sparse_matrix
    DGLGraph.adjacency_matrix
    DGLGraph.adjacency_matrix_scipy
    DGLGraph.incidence_matrix

Using Node/edge features
------------------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.nodes
    DGLGraph.edges
    DGLGraph.ndata
    DGLGraph.edata
    DGLGraph.node_attr_schemes
    DGLGraph.edge_attr_schemes
    DGLGraph.set_n_initializer
    DGLGraph.set_e_initializer
    DGLGraph.local_var
    DGLGraph.local_scope

Computing with DGLGraph
-----------------------

.. autosummary::
    :toctree: ../../generated/

    DGLGraph.register_message_func
    DGLGraph.register_reduce_func
    DGLGraph.register_apply_node_func
    DGLGraph.register_apply_edge_func
    DGLGraph.apply_nodes
    DGLGraph.apply_edges
    DGLGraph.group_apply_edges
    DGLGraph.send
    DGLGraph.recv
    DGLGraph.send_and_recv
    DGLGraph.pull
    DGLGraph.push
    DGLGraph.update_all
    DGLGraph.prop_nodes
    DGLGraph.prop_edges
    DGLGraph.filter_nodes
    DGLGraph.filter_edges
    DGLGraph.to
