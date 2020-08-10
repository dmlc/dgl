.. _apidgl:

dgl
=============================

.. currentmodule:: dgl

Graph Create Ops
-------------------------

.. autosummary::
    :toctree: ../../generated/

    graph
    heterograph
    from_scipy
    from_networkx
    bipartite_from_scipy
    bipartite_from_networkx
    rand_graph
    rand_bipartite
    knn_graph
    segmented_knn_graph

Subgraph Extraction Routines
-------------------------------------

.. autosummary::
    :toctree: ../../generated/

    node_subgraph
    edge_subgraph
    node_type_subgraph
    edge_type_subgraph
    in_subgraph
    out_subgraph

Graph Mutation Routines
---------------------------------

.. autosummary::
    :toctree: ../../generated/

    add_nodes
    add_edges
    remove_nodes
    remove_edges
    add_self_loop
    remove_self_loop
    add_reverse_edges

Graph Transform Routines
----------------------------------

.. autosummary::
    :toctree: ../../generated/

    reverse
    to_bidirected
    to_simple
    to_block
    compact_graphs
    to_hetero
    to_homo
    to_networkx
    line_graph
    khop_graph
    metapath_reachable_graph

Batching and Reading Out
-------------------------------

.. autosummary::
    :toctree: ../../generated/

    batch
    unbatch
    readout_nodes
    readout_edges
    sum_nodes
    sum_edges
    mean_nodes
    mean_edges
    max_nodes
    max_edges
    softmax_nodes
    softmax_edges
    broadcast_nodes
    broadcast_edges
    topk_nodes
    topk_edges

Adjacency Related Routines
-------------------------------

.. autosummary::
    :toctree: ../../generated/

    khop_adj
    laplacian_lambda_max

Propagate Messages by Traversals
------------------------------------------

.. autosummary::
    :toctree: ../../generated/

    prop_nodes
    prop_nodes_bfs
    prop_nodes_topo
    prop_edges
    prop_edges_dfs

Utilities
-----------------------------------------------
.. autosummary::
    :toctree: ../../generated/

    seed
