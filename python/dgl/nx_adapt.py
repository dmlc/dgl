"""Utility functions for networkx adapter."""
from __future__ import absolute_import

from collections import MutableMapping

import networkx as nx
import networkx.convert as convert

class NodeDict(MutableMapping):
    def __init__(self, add_cb, del_cb):
        self._dict = {}
        self._add_cb = add_cb
        self._del_cb = del_cb
    def __setitem__(self, key, val):
        self._add_cb(key)
        self._dict[key] = val
    def __getitem__(self, key):
        return self._dict[key]
    def __delitem__(self, key):
        self._del_cb(key)
        del self._dict[key]
    def __len__(self):
        return len(self._dict)
    def __iter__(self):
        return iter(self._dict)

class AdjOuterDict(MutableMapping):
    def __init__(self, add_cb, del_cb):
        self._dict = {}
        self._add_cb = add_cb
        self._del_cb = del_cb
    def __setitem__(self, key, val):
        val.src = key
        self._dict[key] = val
    def __getitem__(self, key):
        return self._dict[key]
    def __delitem__(self, key):
        for val in self._dict[key]:
            self._del_cb(key, val)
        del self._dict[key]
    def __len__(self):
        return len(self._dict)
    def __iter__(self):
        return iter(self._dict)

class AdjInnerDict(MutableMapping):
    def __init__(self, add_cb, del_cb):
        self._dict = {}
        self.src = None
        self._add_cb = add_cb
        self._del_cb = del_cb
    def __setitem__(self, key, val):
        if self.src is not None and key not in self._dict:
            self._add_cb(self.src, key)
        self._dict[key] = val
    def __getitem__(self, key):
        return self._dict[key]
    def __delitem__(self, key):
        if self.src is not None:
            self._del_cb(self.src, key)
        del self._dict[key]
    def __len__(self):
        return len(self._dict)
    def __iter__(self):
        return iter(self._dict)

class AdjInnerDictFactory(object):
    def __init__(self, cb1, cb2):
        self._cb1 = cb1
        self._cb2 = cb2
    def __call__(self):
        return AdjInnerDict(self._cb1, self._cb2)

def nx_init(obj,
            add_node_cb,
            add_edge_cb,
            del_node_cb,
            del_edge_cb,
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
    obj.adjlist_outer_dict_factory = None
    obj.adjlist_inner_dict_factory = AdjInnerDictFactory(add_edge_cb, del_edge_cb)
    obj.edge_attr_dict_factory = dict

    obj.root_graph = obj
    obj.graph = {}
    obj._node = NodeDict(add_node_cb, del_node_cb)

    obj._adj = AdjOuterDict(add_edge_cb, del_edge_cb)
    obj._pred = dict()
    obj._succ = obj._adj

    if graph_data is not None:
        convert.to_networkx_graph(graph_data, create_using=obj)
    obj.graph.update(attr)
