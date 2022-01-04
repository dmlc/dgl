"""Views of DGLGraph."""
from __future__ import absolute_import

from collections import namedtuple, defaultdict
from collections.abc import MutableMapping

from .base import ALL, DGLError
from . import backend as F

NodeSpace = namedtuple('NodeSpace', ['data'])
EdgeSpace = namedtuple('EdgeSpace', ['data'])

class HeteroNodeView(object):
    """A NodeView class to act as G.nodes for a DGLHeteroGraph."""
    __slots__ = ['_graph', '_typeid_getter']

    def __init__(self, graph, typeid_getter):
        self._graph = graph
        self._typeid_getter = typeid_getter

    def __getitem__(self, key):
        if isinstance(key, slice):
            # slice
            if not (key.start is None and key.stop is None
                    and key.step is None):
                raise DGLError('Currently only full slice ":" is supported')
            nodes = ALL
            ntype = None
        elif isinstance(key, tuple):
            nodes, ntype = key
        elif key is None or isinstance(key, str):
            nodes = ALL
            ntype = key
        else:
            nodes = key
            ntype = None
        ntid = self._typeid_getter(ntype)
        return NodeSpace(data=HeteroNodeDataView(self._graph, ntype, ntid, nodes))

    def __call__(self, ntype=None):
        """Return the nodes."""
        ntid = self._typeid_getter(ntype)
        ret = F.arange(0, self._graph._graph.number_of_nodes(ntid),
                       dtype=self._graph.idtype, ctx=self._graph.device)
        return ret

class HeteroNodeDataView(MutableMapping):
    """The data view class when G.ndata[ntype] is called."""
    __slots__ = ['_graph', '_ntype', '_ntid', '_nodes']

    def __init__(self, graph, ntype, ntid, nodes):
        self._graph = graph
        self._ntype = ntype
        self._ntid = ntid
        self._nodes = nodes

    def __getitem__(self, key):
        if isinstance(self._ntype, list):
            ret = {}
            for (i, ntype) in enumerate(self._ntype):
                value = self._graph._get_n_repr(self._ntid[i], self._nodes).get(key, None)
                if value is not None:
                    ret[ntype] = value
            return ret
        else:
            return self._graph._get_n_repr(self._ntid, self._nodes)[key]

    def __setitem__(self, key, val):
        if isinstance(self._ntype, list):
            assert isinstance(val, dict), \
                'Current HeteroNodeDataView has multiple node types, ' \
                'please passing the node type and the corresponding data through a dict.'

            for (ntype, data) in val.items():
                ntid = self._graph.get_ntype_id(ntype)
                self._graph._set_n_repr(ntid, self._nodes, {key : data})
        else:
            assert isinstance(val, dict) is False, \
                'The HeteroNodeDataView has only one node type. ' \
                'please pass a tensor directly'
            self._graph._set_n_repr(self._ntid, self._nodes, {key : val})

    def __delitem__(self, key):
        if isinstance(self._ntype, list):
            for ntid in self._ntid:
                if self._graph._get_n_repr(ntid, ALL).get(key, None) is None:
                    continue
                self._graph._pop_n_repr(ntid, key)
        else:
            self._graph._pop_n_repr(self._ntid, key)

    def __len__(self):
        assert isinstance(self._ntype, list) is False, \
            'Current HeteroNodeDataView has multiple node types, ' \
            'can not support len().'
        return len(self._graph._node_frames[self._ntid])

    def __iter__(self):
        assert isinstance(self._ntype, list) is False, \
            'Current HeteroNodeDataView has multiple node types, ' \
            'can not be iterated.'
        return iter(self._graph._node_frames[self._ntid])

    def keys(self):
        return self._graph._node_frames[self._ntid].keys()

    def values(self):
        return self._graph._node_frames[self._ntid].values()

    def __repr__(self):
        if isinstance(self._ntype, list):
            ret = defaultdict(dict)
            for (i, ntype) in enumerate(self._ntype):
                data = self._graph._get_n_repr(self._ntid[i], self._nodes)
                for key in self._graph._node_frames[self._ntid[i]]:
                    ret[key][ntype] = data[key]
            return repr(ret)
        else:
            data = self._graph._get_n_repr(self._ntid, self._nodes)
            return repr({key : data[key]
                         for key in self._graph._node_frames[self._ntid]})

class HeteroEdgeView(object):
    """A EdgeView class to act as G.edges for a DGLHeteroGraph."""
    __slots__ = ['_graph']

    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, key):
        if isinstance(key, slice):
            # slice
            if not (key.start is None and key.stop is None
                    and key.step is None):
                raise DGLError('Currently only full slice ":" is supported')
            edges = ALL
            etype = None
        elif key is None:
            edges = ALL
            etype = None
        elif isinstance(key, tuple):
            if len(key) == 3:
                edges = ALL
                etype = key
            else:
                edges = key
                etype = None
        elif isinstance(key, str):
            edges = ALL
            etype = key
        else:
            edges = key
            etype = None
        return EdgeSpace(data=HeteroEdgeDataView(self._graph, etype, edges))

    def __call__(self, *args, **kwargs):
        """Return all the edges."""
        return self._graph.all_edges(*args, **kwargs)

class HeteroEdgeDataView(MutableMapping):
    """The data view class when G.edata[etype] is called."""
    __slots__ = ['_graph', '_etype', '_etid', '_edges']

    def __init__(self, graph, etype, edges):
        self._graph = graph
        self._etype = etype
        self._etid = [self._graph.get_etype_id(t) for t in etype] \
                     if isinstance(etype, list) \
                     else self._graph.get_etype_id(etype)
        self._edges = edges

    def __getitem__(self, key):
        if isinstance(self._etype, list):
            ret = {}
            for (i, etype) in enumerate(self._etype):
                value = self._graph._get_e_repr(self._etid[i], self._edges).get(key, None)
                if value is not None:
                    ret[etype] = value
            return ret
        else:
            return self._graph._get_e_repr(self._etid, self._edges)[key]

    def __setitem__(self, key, val):
        if isinstance(self._etype, list):
            assert isinstance(val, dict), \
                'Current HeteroEdgeDataView has multiple edge types, ' \
                'please pass the edge type and the corresponding data through a dict.'

            for (etype, data) in val.items():
                etid = self._graph.get_etype_id(etype)
                self._graph._set_e_repr(etid, self._edges, {key : data})
        else:
            assert isinstance(val, dict) is False, \
                'The HeteroEdgeDataView has only one edge type. ' \
                'please pass a tensor directly'
            self._graph._set_e_repr(self._etid, self._edges, {key : val})

    def __delitem__(self, key):
        if isinstance(self._etype, list):
            for etid in self._etid:
                if self._graph._get_e_repr(etid, ALL).get(key, None) is None:
                    continue
                self._graph._pop_e_repr(etid, key)
        else:
            self._graph._pop_e_repr(self._etid, key)

    def __len__(self):
        assert isinstance(self._etype, list) is False, \
            'Current HeteroEdgeDataView has multiple edge types, ' \
            'can not support len().'
        return len(self._graph._edge_frames[self._etid])

    def __iter__(self):
        assert isinstance(self._etype, list) is False, \
            'Current HeteroEdgeDataView has multiple edge types, ' \
            'can not be iterated.'
        return iter(self._graph._edge_frames[self._etid])

    def keys(self):
        return self._graph._edge_frames[self._etid].keys()

    def values(self):
        return self._graph._edge_frames[self._etid].values()

    def __repr__(self):
        if isinstance(self._etype, list):
            ret = defaultdict(dict)
            for (i, etype) in enumerate(self._etype):
                data = self._graph._get_e_repr(self._etid[i], self._edges)
                for key in self._graph._edge_frames[self._etid[i]]:
                    ret[key][etype] = data[key]
            return repr(ret)
        else:
            data = self._graph._get_e_repr(self._etid, self._edges)
            return repr({key : data[key]
                         for key in self._graph._edge_frames[self._etid]})
