"""Feature storages for PyTorch tensors."""

import torch
from .base import register_storage_wrapper
from .tensor import BaseTensorStorage
from ..utils import gather_pinned_tensor_rows

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
class PyTorchTensorStorage(BaseTensorStorage):
    """Feature storages for slicing a PyTorch tensor."""
    def fetch(self, indices, device, pin_memory=False):
        device = torch.device(device)
        storage_device_type = self.storage.device.type
        indices_device_type = indices.device.type
        if storage_device_type != 'cuda':
            if indices_device_type == 'cuda':
                if self.storage.is_pinned():
                    return gather_pinned_tensor_rows(self.storage, indices)
                else:
                    raise ValueError(
                        f'Got indices on device {indices.device} whereas the feature tensor '
                        f'is on {self.storage.device}. Please either (1) move the graph '
                        f'to GPU with to() method, or (2) pin the graph with '
                        f'pin_memory_() method.')
            # CPU to CPU or CUDA - use pin_memory and async transfer if possible
            else:
                return _fetch_cpu(indices, self.storage, self.storage.shape[1:], device,
                                  pin_memory)
        else:
            # CUDA to CUDA or CPU
            return _fetch_cuda(indices, self.storage, device)
