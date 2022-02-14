"""Feature storages for PyTorch tensors."""

import torch
from .base import FeatureStorage, register_storage_wrapper

def _fetch_cpu(indices, tensor, feature_shape, device, pin_memory):
    result = torch.empty(
        indices.shape[0], *feature_shape, dtype=tensor.dtype,
        pin_memory=pin_memory)
    torch.index_select(tensor, 0, indices, out=result)
    result = result.to(device, non_blocking=True)
    return result

def _fetch_cuda(indices, tensor, device):
    return torch.index_select(tensor, 0, indices).to(device)

@register_storage_wrapper(torch.Tensor)
class TensorStorage(FeatureStorage):
    """Feature storages for slicing a PyTorch tensor."""
    def __init__(self, tensor):
        self.storage = tensor
        self.feature_shape = tensor.shape[1:]
        self.is_cuda = (tensor.device.type == 'cuda')

    def fetch(self, indices, device, pin_memory=False):
        device = torch.device(device)
        if not self.is_cuda:
            # CPU to CPU or CUDA - use pin_memory and async transfer if possible
            return _fetch_cpu(indices, self.storage, self.feature_shape, device, pin_memory)
        else:
            # CUDA to CUDA or CPU
            return _fetch_cuda(indices, self.storage, device)
