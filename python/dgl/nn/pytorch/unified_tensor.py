"""Unified Tensor."""
from dgl._ffi.runtime_ctypes import DGLContext
import torch as th
from ...backend import pytorch as F
import dgl.ndarray as nd


class UnifiedTensor: #UnifiedTensor
    '''Class for storing unified tensor.

    Parameters
    ----------
    input : torch tensor
        Torch tensor which we want to convert into the
        unified tensor.
    device : th.device
        Device to create the mapping of the unified tensor.
    '''

    def __init__(self, input, device):
        if device.type != 'cuda':
            raise ValueError("Target device must be a cuda device")

        self._input = input
        self._array = F.zerocopy_to_dgl_ndarray(self._input)
        self._device = device

        # Pin & map the host memory space
        self._array.pin_memory_(nd.gpu(self._device.index))

    def __len__(self):
        return len(self._array)

    def __repr__(self):
        pass

    def __getitem__(self, key):
        return F.zerocopy_from_dgl_ndarray(nd.uvm_gather(self._array, F.zerocopy_to_dgl_ndarray(key)))

    @property
    def shape(self):
        """Shape of this tensor"""
        return self._array.shape

    @property
    def dtype(self):
        """Type of this tensor"""
        return self._array.dtype

    @property
    def device(self):
        """Device of this tensor"""
        return self._device