"""Unified Tensor."""
from .. import backend as F
from .._ffi.function import _init_api
from .. import utils


class UnifiedTensor: #UnifiedTensor
    '''Class for storing unified tensor. Declaration of
    UnifiedTensor automatically pins the input tensor.
    Upon a successful declaration of UnifiedTensor, the
    target GPU device will have the address mapping of the
    input CPU tensor for zero-copy (direct) access over
    external interconnects (e.g., PCIe).

    Parameters
    ----------
    input : Tensor
        Tensor which we want to convert into the
        unified tensor.
    device : device
        GPU to create the address mapping of the input CPU tensor.

    Examples
    --------
    With a given CPU tensor ``feats``, a new UnifiedTensor targetting a default
    GPU can be created as follows:

    >>> feats = torch.rand((128,128))
    >>> feats = dgl.contrib.UnifiedTensor(feats, device=torch.device('cuda'))

    Now, the elements of the new tensor ``feats`` can be accessed with ``[]``
    indexing. The context of the index tensor is a switch to trigger the
    zero-copy access from GPU. For example, to use the ordinary CPU-based
    data access, one can use the following method:

    >>> idx = torch.Tensor([0,1,2])
    >>> output = feats[idx]

    Now, to use GPU to do a zero-copy access, do this:

    >>> idx = torch.Tensor([0,1,2]).to('cuda')
    >>> output = feats[idx]

    For the multi-GPU operation, to allow multiple GPUs to access the original CPU tensor
    ``feats`` using UnifiedTensor, one can do the following:

    >>> feats = torch.rand((128,128))
    >>> feats_gpu0 = dgl.contrib.UnifiedTensor(feats, device=torch.device('cuda:0'))
    >>> feats_gpu1 = dgl.contrib.UnifiedTensor(feats, device=torch.device('cuda:1'))
    >>> feats_gpu2 = dgl.contrib.UnifiedTensor(feats, device=torch.device('cuda:2'))

    Now, the ``cuda:0``, ``cuda:1``, and ``cuda:2`` devices will be able to access the
    identical tensor located in the CPU memory using ``feats_gpu0``, ``feats_gpu1``, and ``feats_gpu2`` tensors, respectively.
    
    One can simply use following operations to slice the sub tensors into different GPU devices directly.

    >>> feats_idx_gpu0 = torch.randint(128, 16, device='cuda:0')
    >>> feats_idx_gpu1 = torch.randint(128, 16, device='cuda:1')
    >>> feats_idx_gpu2 = torch.randint(128, 16, device='cuda:2')
    
    >>> sub_feat_gpu0 = feats_gpu0[feats_idx_gpu0]
    >>> sub_feat_gpu1 = feats_gpu1[feats_idx_gpu1]
    >>> sub_feat_gpu2 = feats_gpu2[feats_idx_gpu2]

    ``feats_gpu2`` tensors, respectively.
    '''

    def __init__(self, input, device):
        if F.device_type(device) != 'cuda':
            raise ValueError("Target device must be a cuda device")
        if F.device_type(F.context(input)) != 'cpu':
            raise ValueError("Input tensor must be a cpu tensor")

        self._input = input
        self._array = F.zerocopy_to_dgl_ndarray(self._input)
        self._device = device

        self._array.pin_memory_()

    def __len__(self):
        return len(self._array)

    def __repr__(self):
        return self._input.__repr__()

    def __getitem__(self, key):
        '''Perform zero-copy access from GPU if the context of
        the key is cuda. Otherwise, just safely fallback to the
        backend specific indexing scheme.

        Parameters
        ----------
        key : Tensor
            Tensor which contains the index ids
        '''
        if F.device_type(F.context(key)) != 'cuda':
            return self._input[key]
        else:
            return F.zerocopy_from_dgl_ndarray(
                    _CAPI_DGLIndexSelectCPUFromGPU(self._array,
                                F.zerocopy_to_dgl_ndarray(key)))

    def __setitem__(self, key, val):
        self._input[key] = val

    def __del__(self):
        if hasattr(self, '_array') and self._array != None:
            self._array.unpin_memory_()
            self._array = None

        if hasattr(self, '_input'):
            self._input = None

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

_init_api("dgl.ndarray.uvm", __name__)
