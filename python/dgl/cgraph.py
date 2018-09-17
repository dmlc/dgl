from __future__ import absolute_import

from ._ffi.function import _init_api
from . import backend as F
from . import utils

class DGLGraph(object):
    def __init__(self):
        self._handle = _CAPI_DGLGraphCreate()

    def __del__(self):
        _CAPI_DGLGraphFree(self._handle)

    def add_nodes(self, num):
        _CAPI_DGLGraphAddVertices(self._handle, num);

    def add_edge(self, u, v):
        _CAPI_DGLGraphAddEdge(self._handle, u, v);

    def add_edges(self, u, v):
        u = utils.Index(u)
        v = utils.Index(v)
        u_array = F.asdglarray(u.totensor())
        v_array = F.asdglarray(v.totensor())
        _CAPI_DGLGraphAddEdges(self._handle, u_array, v_array)

    def clear(self):
        _CAPI_DGLGraphClear(self._handle)

    def number_of_nodes(self):
        return _CAPI_DGLGraphNumVertices(self._handle)

    def number_of_edges(self):
        return _CAPI_DGLGraphNumEdges(self._handle)

    def has_vertex(self, vid):
        return _CAPI_DGLGraphHasVertex(self._handle, vid)

    def has_vertices(self, vids):
        vids = utils.Index(vids)
        vid_array = F.asdglarray(vids.totensor())
        return _CAPI_DGLGraphHasVertices(self._handle, vid_array)

    def has_edge(self, u, v):
        return _CAPI_DGLGraphHasEdge(self._handle, u, v)

    def has_edges(self, u, v):
        u = utils.Index(u)
        v = utils.Index(v)
        u_array = F.asdglarray(u.totensor())
        v_array = F.asdglarray(v.totensor())
        return _CAPI_DGLGraphHasEdges(self._handle, u_array, v_array)

    def predecessors(self, v, radius=1):
        return _CAPI_DGLGraphPredecessors(self._handle, v, radius)

    def successors(self, v, radius=1):
        return _CAPI_DGLGraphSuccessors(self._handle, v, radius)

    def edge_id(self, u, v):
        return _CAPI_DGLGraphEdgeId(self._handle, u, v)

    def edge_ids(self, u, v):
        u = utils.Index(u)
        v = utils.Index(v)
        u_array = F.asdglarray(u.totensor())
        v_array = F.asdglarray(v.totensor())
        return _CAPI_DGLGraphEdgeIds(self._handle, u_array, v_array)

    def inedges(self, v):
        if isinstance(v, int):
            pair = _CAPI_DGLGraphInEdges_1(self._handle, v)
        else:
            v = utils.Index(v)
            v_array = F.asdglarray(v.totensor())
            pair = _CAPI_DGLGraphInEdges_2(self._handle, v_array)
        return pair(0), pair(1)

    def outedges(self, v):
        if isinstance(v, int):
            pair = _CAPI_DGLGraphOutEdges_1(self._handle, v)
        else:
            v = utils.Index(v)
            v_array = F.asdglarray(v.totensor())
            pair = _CAPI_DGLGraphOutEdges_2(self._handle, v_array)
        return pair(0), pair(1)

    def indegree(self, v):
        return _CAPI_DGLGraphInDegree(self._handle, v)

    def indegrees(self, v):
        v = utils.Index(v)
        v_array = F.asdglarray(v.totensor())
        return _CAPI_DGLGraphInDegrees(self._handle, v_array)

    def outdegree(self, v):
        return _CAPI_DGLGraphOutDegree(self._handle, v)

    def outdegrees(self, v):
        v = utils.Index(v)
        v_array = F.asdglarray(v.totensor())
        return _CAPI_DGLGraphOutDegrees(self._handle, v_array)

    def edges(self, sorted=False):
        pair = _CAPI_DGLGraphEdges(self._handle, sorted)
        return pair(0), pair(1)

_init_api("dgl.cgraph")
