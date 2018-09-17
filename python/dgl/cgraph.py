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
        _CAPI_DGLGraphAddEdges(
                self._handle,
                u_array,
                v_array)

    def number_of_nodes(self):
        return _CAPI_DGLGraphNumVertices(self._handle)

    def number_of_edges(self):
        return _CAPI_DGLGraphNumEdges(self._handle)

_init_api("dgl.cgraph")
