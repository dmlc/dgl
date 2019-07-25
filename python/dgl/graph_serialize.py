from __future__ import absolute_import

import ctypes
import numpy as np
import networkx as nx
import scipy

from .graph import DGLGraph
from ._ffi.object import ObjectBase, register_object
# from ._ffi.base import c_array
from . import _api_internal
from ._ffi.function import _init_api
from .base import DGLError
from . import backend as F
from . import utils

_init_api("dgl.graph_serialize")


@register_object("graph_serialize.GraphData")
class GraphData(ObjectBase):
    @staticmethod
    def create(g: DGLGraph):
        ghandle = g._graph
        node_tensors = dict()
        edge_tensors = dict()
        for key, value in g.ndata.items():
            node_tensors[key] = F.zerocopy_to_dgl_ndarray(value)
        for key, value in g.edata.items():
            edge_tensors[key] = F.zerocopy_to_dgl_ndarray(value)
        return _CAPI_MakeGraphData(ghandle, node_tensors, edge_tensors)

    def getGraph(self):
        ghandle = _CAPI_GDataGraphHandle(self)
        g = DGLGraph(graph_data=ghandle, readonly=True)
        node_tensors_items = _CAPI_GDataNodeTensors(self).items()
        edge_tensors_items = _CAPI_GDataEdgeTensors(self).items()
        for k, v in node_tensors_items:
            g.ndata[k] = F.zerocopy_from_dgl_ndarray(_api_internal._ValueGet(v))
        for k, v in edge_tensors_items:
            g.edata[k] = F.zerocopy_from_dgl_ndarray(_api_internal._ValueGet(v))
        return g


def save_graphs(filename, g_list):
    gdata_list = [GraphData.create(g) for g in g_list]
    _CAPI_DGLSaveGraphs(filename, gdata_list)

def load_graphs(filename, idx_list=None):
    if idx_list is None:
        idx_list = []
    gdata_list = _CAPI_DGLLoadGraphs(filename, idx_list)
    return [gdata.getGraph() for gdata in gdata_list]

    

# _CAPI_MakeGraphData(g._graph)
# p