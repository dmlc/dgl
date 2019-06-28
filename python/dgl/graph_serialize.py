from __future__ import absolute_import

import ctypes
import numpy as np
import networkx as nx
import scipy

from ._ffi.runtime_ctypes import DGLArrayHandle, DGLArray
# from ._ffi.base import c_array
from ._ffi.function import _init_api
from .base import DGLError
from . import backend as F
from . import utils
from .ndarray import from_dlpack, NDArrayBase

# from ._ffi._ctypes.function import _make_array

_init_api("dgl.graph_serialize")

GraphIndexHandle = ctypes.c_void_p


# class DGLGraphSerialize(ctypes.Structure):
#     __fields__ = [("g_handle", GraphIndexHandle),
#                   ("num_node_feats", ctypes.c_uint32),
#                   ("num_edge_feats", ctypes.c_uint32),
#                   ("node_names", ctypes.c_char_p),
#                   ("node_feats", DGLArrayHandle),
#                   ("edge_names", ctypes.c_char_p),
#                   ("edge_feats", DGLArrayHandle)
#                   ]
class DGLGraphSerialize(ctypes.Structure):
    __fields__ = [("g_handle", GraphIndexHandle),
                  ("num_node_feats", ctypes.c_uint32),
                  ("num_edge_feats", ctypes.c_uint32),
                  ("node_names", ctypes.POINTER(ctypes.c_char_p)),
                  ("node_feats", ctypes.POINTER(DGLArrayHandle)),
                  ("edge_names", ctypes.POINTER(ctypes.c_char_p)),
                  ("edge_feats", ctypes.POINTER(DGLArrayHandle)),
                  ]


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

        g_list.append(g)
    return g_list


def convert_list_tensor(tensor_list):
    return [from_dlpack(F.zerocopy_to_dlpack(tensor)).handle for tensor in tensor_list]


def convert_str_list(str_list):
    return [str.encode('utf-8') for str in str_list]


def c_array(type, list):
    aaa = (type * len(list))()
    for idx, item in enumerate(list):
        aaa[idx] = item
    return ctypes.cast(aaa, ctypes.POINTER(type))


def saveDGLGraph():
    g_list = construct_graph(5)

    args_list = (DGLGraphSerialize * len(g_list))()
    args_list = ctypes.cast(args_list, ctypes.POINTER(DGLGraphSerialize))

    for idx, g in enumerate(g_list):
        arg = DGLGraphSerialize()
        print(g)
        print(g._graph.handle)
        arg.g_handle = g._graph.handle
        arg.num_node_feats = ctypes.c_uint32(len(g.ndata))
        arg.num_edge_feats = ctypes.c_uint32(len(g.edata))

        # arg.node_names = ctypes.POINTER()
        # for name in convert_str_list(g.ndata.keys()):

        arg.node_names = c_array(ctypes.c_char_p, convert_str_list(g.ndata.keys()))
        # print(type(arg.node_names))
        arg.node_feats = c_array(DGLArrayHandle, convert_list_tensor(g.ndata.values()))
        arg.edge_names = c_array(ctypes.c_char_p, convert_str_list(g.edata.keys()))
        arg.edge_feats = c_array(DGLArrayHandle, convert_list_tensor(g.edata.values()))
        print(arg.node_names[0])
        print(arg.node_feats)

        args_list[idx] = arg
    #
    # args_list = c_array(ctypes.POINTER(DGLGraphSerialize), args_list)
    # list_ptr =
    inputs = ctypes.cast(args_list, ctypes.c_void_p)
    import ipdb;
    ipdb.set_trace()
    print(list_ptr[0].value)
    #
    _CAPI_DGLSaveGraphs(inputs, len(args_list),
                        "/tmp/test.bin")


def loadDGLGraph():
    idx_list = [1, 3]
    idx_input = ctypes.cast(c_array(ctypes.c_uint32, idx_list), ctypes.c_void_p)

    ret = _CAPI_DGLLoadGraphs("/tmp/test.bin", idx_input)
    g_list = ctypes.cast(ret, ctypes.POINTER(DGLGraphSerialize))
    print(g_list)
