"""API creating NCCL communicators."""

from .. import backend as F
from .. import ndarray
from .. import utils
from .._ffi.function import _init_api

class UniqueId(object):
    def __init__(self):
        """ Create an object reference the current NCCL unique id.
        """
        self._handle = _CAPI_DGLNCCLGetUniqueId()

    def get(self):
        return self._handle
    
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

    def get(self):
        return self._handle

_init_api("dgl.cuda.nccl")
