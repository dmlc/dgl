from __future__ import absolute_import

import ctypes
import numpy as np
import networkx as nx
import scipy

from .graph import DGLGraph
from ._ffi.object import ObjectBase, register_object
from ._ffi.runtime_ctypes import DGLArrayHandle, DGLArray
# from ._ffi.base import c_array
from ._ffi.function import _init_api
from .base import DGLError
from . import backend as F
from . import utils
from .ndarray import from_dlpack, NDArrayBase

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


# register_object("Value")(NDArrayBase)
# from ._ffi._ctypes.function import _make_array
#
def construct_graph(n):
    from .graph import DGLGraph
    g_list = []
    for i in range(n):
        g = DGLGraph()
        g.add_nodes(10)
        g.add_edges(1, 2)
        g.add_edges(3, 2)
        g.add_edges(3, 3)

        import torch as th

        g.edata['e1'] = th.ones(3, 5)
        g.edata['e2'] = th.zeros(3, 5)
        g.ndata['n1'] = th.ones(10, 2)
        g.readonly()
        g_list.append(g)
    return g_list


#
def aaa():
    g_list = construct_graph(3)

    # g = g_list[0]

    g_data = []
    for g in g_list:
        g_data.append(GraphData.create(g))

    print(g_data)
    print(g_data[0])
    _CAPI_DGLSaveGraphs("/tmp/test.bin", g_data)


aaa()

# _CAPI_MakeGraphData(g._graph)
