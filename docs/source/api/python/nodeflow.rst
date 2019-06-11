.. _apinodeflow:

NodeFlow -- Graph sampled from a large graph
============================================

.. currentmodule:: dgl
.. autoclass:: NodeFlow

Querying graph structure
------------------------

.. autosummary::
    :toctree: ../../generated/

    NodeFlow.num_layers
    NodeFlow.num_blocks
    NodeFlow.layer_size
    NodeFlow.block_size
    NodeFlow.layer_in_degree
    NodeFlow.layer_out_degree
    NodeFlow.layer_nid
    NodeFlow.layer_parent_nid
    NodeFlow.block_eid
    NodeFlow.block_parent_eid
    NodeFlow.block_edges

Converting to other format
-------------------------------

.. autosummary::
    :toctree: ../../generated/

    NodeFlow.block_adjacency_matrix
    NodeFlow.block_incidence_matrix

Using Node/edge features
------------------------

.. autosummary::
    :toctree: ../../generated/

    NodeFlow.layers
    NodeFlow.blocks
    NodeFlow.set_n_initializer
    NodeFlow.set_e_initializer
    NodeFlow.node_attr_schemes
    NodeFlow.edge_attr_schemes

Mapping between NodeFlow and parent graph
-----------------------------------------
.. autosummary::
    :toctree: ../../generated/

    NodeFlow.map_to_parent_nid
    NodeFlow.map_to_parent_eid
    NodeFlow.map_from_parent_nid


Synchronize features between NodeFlow and parent graph
------------------------------------------------------
.. autosummary::
    :toctree: ../../generated/

    NodeFlow.copy_from_parent
    NodeFlow.copy_to_parent

Computing with NodeFlow
-----------------------

.. autosummary::
    :toctree: ../../generated/

    NodeFlow.register_message_func
    NodeFlow.register_reduce_func
    NodeFlow.register_apply_node_func
    NodeFlow.register_apply_edge_func
    NodeFlow.apply_layer
    NodeFlow.apply_block
    NodeFlow.block_compute
    NodeFlow.prop_flow
