"""Utility functions for networkx adapter."""
from __future__ import absolute_import

from collections import MutableMapping

import networkx as nx
import networkx.convert as convert

class NodeDict(MutableMapping):
    def __init__(self, cb):
        self._dict = {}
        self._cb = cb
    def __setitem__(self, key, val):
        if isinstance(val, AdjInnerDict):
            # This node dict is used as adj_outer_list
            val.src = key
        elif key not in self._dict:
            self._cb(key)
        self._dict[key] = val
    def __getitem__(self, key):
        return self._dict[key]
    def __delitem__(self, key):
        del self._dict[key]
    def __len__(self):
        return len(self._dict)
    def __iter__(self):
        return iter(self._dict)

class AdjInnerDict(MutableMapping):
    def __init__(self, cb):
        self._dict = {}
        self.src = None
        self._cb = cb
    def __setitem__(self, key, val):
        if key not in self._dict:
            self._cb(self.src, key)
        self._dict[key] = val
    def __getitem__(self, key):
        return self._dict[key]
    def __delitem__(self, key):
        del self._dict[key]
    def __len__(self):
        return len(self._dict)
    def __iter__(self):
        return iter(self._dict)

def nx_init(obj,
            add_node_cb,
            add_edge_cb,
            graph_data,
            **attr):
    """Init the object to be compatible with networkx's DiGraph.

    Parameters
    ----------
    obj : any
        The object to be init.
    add_node_cb : callable
        The callback function when node is added.
    add_edge_cb : callable
        The callback function when edge is added.
    graph_data : graph data
        Data to initialize graph. Same as networkx's semantics.
    attr : keyword arguments, optional
        Attributes to add to graph as key=value pairs.
    """
    # The following codes work for networkx 2.1.
    obj.node_dict_factory = ndf = lambda : NodeDict(add_node_cb)
    obj.adjlist_outer_dict_factory = None
    obj.adjlist_inner_dict_factory = lambda : AdjInnerDict(add_edge_cb)
    obj.edge_attr_dict_factory = dict

    obj.root_graph = obj
    obj.graph = {}
    obj._node = ndf()

    obj._adj = ndf()
    obj._pred = ndf()
    obj._succ = obj._adj

    if graph_data is not None:
        convert.to_networkx_graph(graph_data, create_using=obj)
    obj.graph.update(attr)
