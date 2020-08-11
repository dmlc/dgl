.. _apigraph:

dgl.DGLHeteroGraph
=====================================================

.. currentmodule:: dgl
.. autoclass:: DGLHeteroGraph

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
    DGLHeteroGraph.has_nodes
    DGLHeteroGraph.has_edges_between
    DGLHeteroGraph.predecessors
    DGLHeteroGraph.successors
    DGLHeteroGraph.edge_ids
    DGLHeteroGraph.find_edges
    DGLHeteroGraph.in_edges
    DGLHeteroGraph.out_edges
    DGLHeteroGraph.all_edges
    DGLHeteroGraph.in_degrees
    DGLHeteroGraph.out_degrees

Querying and manipulating sparse format
---------------------------------------

.. autosummary::
    :toctree: ../../generated/

    DGLHeteroGraph.format_in_use
    DGLHeteroGraph.restrict_format
    DGLHeteroGraph.to_format

Querying and manipulating index data type
-----------------------------------------

.. autosummary::
    :toctree: ../../generated/

    DGLHeteroGraph.idtype
    DGLHeteroGraph.long
    DGLHeteroGraph.int

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
    DGLHeteroGraph.send_and_recv
    DGLHeteroGraph.pull
    DGLHeteroGraph.push
    DGLHeteroGraph.update_all
    DGLHeteroGraph.multi_update_all
    DGLHeteroGraph.prop_nodes
    DGLHeteroGraph.prop_edges
    DGLHeteroGraph.filter_nodes
    DGLHeteroGraph.filter_edges
    DGLHeteroGraph.to
