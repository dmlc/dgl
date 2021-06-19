"""Define utility functions for shared memory."""

from .. import backend as F
from .._ffi.ndarray import empty_shared_mem
from .. import ndarray as nd

DTYPE_DICT = F.data_type_dict
DTYPE_DICT = {DTYPE_DICT[key]:key for key in DTYPE_DICT}

def _get_ndata_path(graph_name, ndata_name):
    return "/" + graph_name + "_node_" + ndata_name

def _get_edata_path(graph_name, edata_name):
    return "/" + graph_name + "_edge_" + edata_name

def _to_shared_mem(arr, name):
    dlpack = F.zerocopy_to_dlpack(arr)
    dgl_tensor = nd.from_dlpack(dlpack)
    new_arr = empty_shared_mem(name, True, F.shape(arr), DTYPE_DICT[F.dtype(arr)])
    dgl_tensor.copyto(new_arr)
    dlpack = new_arr.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)
