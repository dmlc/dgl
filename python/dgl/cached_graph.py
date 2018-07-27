"""High-performance graph structure query component.

TODO: Currently implemented by networkx. Should replace with more efficient
solution later.
"""

import networkx as nx

class CachedGraph:
    def __init__(self):
        pass

    def add_node(self, u):
        # TODO: make sure the node id is equal to row id in storage.
        pass

    def add_edge(self, u, v):
        # TODO: one-many, many-one, many-many
        # TODO: make sure the edge id is equal to row id in storage.
        pass

    def get_edge_id(self, u, v):
        # TODO: one-many, many-one, many-many
        # TODO: return a tensor of edge ids.
        pass

    def in_edges(self, v):
        # TODO: return two tensors, src and dst.
        pass

    def out_edges(self, u):
        # TODO: return two tensors, src and dst.
        pass

    def edges(self):
        # TODO: return two tensors, src and dst.
        pass
