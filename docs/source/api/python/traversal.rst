dgl.traversal
===============

.. automodule:: dgl.traversal

Graph traversal algorithms implemented as python generators, which returns the visited set
of nodes or edges at each iteration. The naming convention
is ``<algorithm>_[nodes|edges]_generator``. An example usage is as follows.

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
