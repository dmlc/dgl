.. _apigraphstore:

Graph Store -- Graph for multi-processing and distributed training
==================================================================

.. currentmodule:: dgl.contrib.graph_store
.. autoclass:: SharedMemoryDGLGraph

Querying the distributed setting
--------------------------------

.. autosummary::
    :toctree: ../../generated/

    SharedMemoryDGLGraph.num_workers
    SharedMemoryDGLGraph.worker_id
    SharedMemoryDGLGraph.destroy

Using Node/edge features
------------------------

.. autosummary::
    :toctree: ../../generated/

    SharedMemoryDGLGraph.init_ndata
    SharedMemoryDGLGraph.init_edata

Computing with Graph store
--------------------------

.. autosummary::
    :toctree: ../../generated/

    SharedMemoryDGLGraph.apply_nodes
    SharedMemoryDGLGraph.apply_edges
    SharedMemoryDGLGraph.group_apply_edges
    SharedMemoryDGLGraph.recv
    SharedMemoryDGLGraph.send_and_recv
    SharedMemoryDGLGraph.pull
    SharedMemoryDGLGraph.push
    SharedMemoryDGLGraph.update_all

Construct a graph store
-----------------------

.. autofunction:: dgl.contrib.graph_store.create_graph_store_server
.. autofunction:: dgl.contrib.graph_store.create_graph_from_store
