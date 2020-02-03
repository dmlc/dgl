dgl.propagate
===============

.. automodule:: dgl.propagate

Propagate messages and perform computation following graph traversal order. ``prop_nodes_XXX``
calls traversal algorithm ``XXX`` and triggers :func:`~DGLGraph.pull()` on the visited node
set at each iteration. ``prop_edges_YYY`` applies traversal algorithm ``YYY`` and triggers
:func:`~DGLGraph.send_and_recv()` on the visited edge set at each iteration.

.. autosummary::
    :toctree: ../../generated/

    prop_nodes
    prop_edges
    prop_nodes_bfs
    prop_nodes_topo
    prop_edges_dfs
