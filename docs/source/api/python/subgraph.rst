.. _apisubgraph:

DGLSubGraph -- Class for subgraph data structure
================================================

.. currentmodule:: dgl.subgraph
.. autoclass:: DGLSubGraph

Mapping between subgraph and parent graph
-----------------------------------------
.. autosummary::
    :toctree: ../../generated/

    DGLSubGraph.parent_nid
    DGLSubGraph.parent_eid
    DGLSubGraph.map_to_subgraph_nid  

Synchronize features between subgraph and parent graph
------------------------------------------------------
.. autosummary::
    :toctree: ../../generated/

    DGLSubGraph.copy_from_parent
    DGLSubGraph.copy_to_parent
