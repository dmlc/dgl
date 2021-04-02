"""API creating NCCL communicators."""

from .. import backend as F
from .. import ndarray
from .. import utils
from .._ffi.function import _init_api

_COMM_MODES_MAP = {
    'remainder': 0
}

class UniqueId(object):
    def __init__(self, id_str=None):
        """ Create an object reference the current NCCL unique id.
        """
        if id_str:
            self._handle = _CAPI_DGLNCCLUniqueIdFromString(id_str);
        else:
            self._handle = _CAPI_DGLNCCLGetUniqueId()

    def get(self):
        return self._handle

    def __str__(self):
        return _CAPI_DGLNCCLUniqueIdToString(self._handle)
    
class Communicator(object):
    def __init__(self, size, rank, unique_id):
        """ Create a new NCCL communicator.
            
            Parameters
            ----------
            size : int
                The number of processes in the communicator.
            rank : int
                The rank of the current process in the communicator.
            unique_id : NCCLUniqueId
                The unique id of the root process (rank=0).
        """
        self._handle = _CAPI_DGLNCCLCreateComm(size, rank, unique_id.get())
        assert rank < size
        self._rank = rank
        self._size = size

    def sparse_all_to_all_push(self, idx, value, mode):
        mode_id = _COMM_MODES_MAP[mode]

        out_idx, out_value = _CAPI_DGLNCCLSparseAllToAllPush(
            self.get(), F.zerocopy_to_dgl_ndarray(idx),
            F.zerocopy_to_dgl_ndarray(value),
            mode_id)
        return (F.zerocopy_from_dgl_ndarray(out_idx),
            F.zerocopy_from_dgl_ndarray(out_value))

    def sparse_all_to_all_pull(self, req_idx, value, mode):
        mode_id = _COMM_MODES_MAP[mode]

        out_value = _CAPI_DGLNCCLSparseAllToAllPull(
            self.get(), F.zerocopy_to_dgl_ndarray(req_idx),
            F.zerocopy_to_dgl_ndarray(value),
            mode_id)
        return F.zerocopy_from_dgl_ndarray(out_value)

    def get(self):
        return self._handle

    def rank(self):
        return self._rank

    def size(self):
        return self._size

_init_api("dgl.cuda.nccl")
