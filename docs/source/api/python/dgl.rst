.. _apidgl:

dgl
=============================

.. currentmodule:: dgl
.. automodule:: dgl

.. _api-graph-create-ops:

Graph Create Ops
-------------------------

Operators for constructing :class:`DGLGraph` from raw data formats.

.. autosummary::
    :toctree: ../../generated/

    graph
    heterograph
    from_cugraph
    from_scipy
    from_networkx
    bipartite_from_scipy
    bipartite_from_networkx
    rand_graph
    rand_bipartite
    knn_graph
    segmented_knn_graph
    radius_graph
    create_block
    block_to_graph
    merge

.. _api-subgraph-extraction:

Subgraph Extraction Ops
-------------------------------------

Operators for extracting and returning subgraphs.

.. autosummary::
    :toctree: ../../generated/

    node_subgraph
    edge_subgraph
    node_type_subgraph
    edge_type_subgraph
    in_subgraph
    out_subgraph
    khop_in_subgraph
    khop_out_subgraph

.. _api-transform:

Graph Transform Ops
----------------------------------

Operators for generating new graphs by manipulating the structure of the existing ones.

.. autosummary::
    :toctree: ../../generated/

    add_edges
    add_nodes
    add_reverse_edges
    add_self_loop
    adj_product_graph
    adj_sum_graph
    compact_graphs
    khop_adj
    khop_graph
    knn_graph
    laplacian_lambda_max
    line_graph
    metapath_reachable_graph
    metis_partition
    metis_partition_assignment
    norm_by_dst
    partition_graph_with_halo
    radius_graph
    remove_edges
    remove_nodes
    remove_self_loop
    reorder_graph
    reverse
    segmented_knn_graph
    sort_csr_by_tag
    sort_csc_by_tag
    to_bidirected
    to_bidirected_stale
    to_block
    to_cugraph
    to_double
    to_float
    to_half
    to_heterogeneous
    to_homogeneous
    to_networkx
    to_simple
    to_simple_graph

.. _api-positional-encoding:

Graph Positional Encoding Ops:
-----------------------------------------

Operators for generating positional encodings of each node.

.. autosummary::
    :toctree: ../../generated

    random_walk_pe
    lap_pe
    double_radius_node_labeling
    shortest_dist
    svd_pe

.. _api-partition:

Graph Partition Utilities
-------------------------
.. autosummary::
    :toctree: ../../generated/

    metis_partition
    metis_partition_assignment
    partition_graph_with_halo

.. _api-batch:

Batching and Reading Out Ops
-------------------------------

Operators for batching multiple graphs into one for batch processing and
operators for computing graph-level representation for both single and batched graphs.

.. autosummary::
    :toctree: ../../generated/

    batch
    unbatch
    slice_batch
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

Adjacency Related Utilities
-------------------------------

Utilities for computing adjacency matrix and Lapacian matrix.

.. autosummary::
    :toctree: ../../generated/

    khop_adj
    laplacian_lambda_max

Graph Traversal & Message Propagation
------------------------------------------

DGL implements graph traversal algorithms implemented as python generators,
which returns the visited set of nodes or edges (in ID tensor) at each iteration.
The naming convention is ``<algorithm>_[nodes|edges]_generator``.
An example usage is as follows.

.. code:: python

    g = ...  # some DGLGraph
    for nodes in dgl.bfs_nodes_generator(g, 0):
        do_something(nodes)

.. autosummary::
    :toctree: ../../generated/

    bfs_nodes_generator
    bfs_edges_generator
    topological_nodes_generator
    dfs_edges_generator
    dfs_labeled_edges_generator

DGL provides APIs to perform message passing following graph traversal order. ``prop_nodes_XXX``
calls traversal algorithm ``XXX`` and triggers :func:`~DGLGraph.pull()` on the visited node
set at each iteration. ``prop_edges_YYY`` applies traversal algorithm ``YYY`` and triggers
:func:`~DGLGraph.send_and_recv()` on the visited edge set at each iteration.

.. autosummary::
    :toctree: ../../generated/

    prop_nodes
    prop_nodes_bfs
    prop_nodes_topo
    prop_edges
    prop_edges_dfs

Homophily Measures
-------------------------

Utilities for measuring homophily of a graph

.. autosummary::
    :toctree: ../../generated/

    edge_homophily
    node_homophily
    linkx_homophily
    adjusted_homophily

Label Informativeness Measures
-------------------------

Utilities for measuring label informativeness of a graph

.. autosummary::
    :toctree: ../../generated/

    edge_label_informativeness
    node_label_informativeness

Utilities
-----------------------------------------------

Other utilities for controlling randomness, saving and loading graphs, setting and getting runtime configurations, functions that applies
the same function to every elements in a container, etc.

.. autosummary::
    :toctree: ../../generated/

    seed
    save_graphs
    load_graphs
    apply_each
    use_libxsmm
    is_libxsmm_enabled
