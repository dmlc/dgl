from collections.abc import Mapping
import torch
from ..utils import recursive_apply, recursive_apply_pair
from .base import FeatureStorage

def _fetch_cpu(indices, tensor, feature_shape, device, pin_memory):
    result = torch.empty(
        indices.shape[0], *feature_shape, dtype=tensor.dtype,
        pin_memory=pin_memory)
    torch.index_select(tensor, 0, indices, out=result)
    result = result.to(device, non_blocking=True)
    return result

def _fetch_cuda(indices, tensor, device):
    return torch.index_select(tensor, 0, indices).to(device)

class TensorStorage(FeatureStorage):
    def __init__(self, tensor):
        self.tensor = tensor
        self.feature_shape = recursive_apply(tensor, lambda x: x.shape[1:])
        if isinstance(tensor, Mapping):
            self.is_cuda = (next(iter(tensor.values())).device.type == 'cuda')
        else:
            self.is_cuda = (tensor.device.type == 'cuda')

    def fetch(self, indices, device, pin_memory=False):
        device = torch.device(device)
        if not self.is_cuda:
            # CPU to CPU or CUDA - use pin_memory and async transfer if possible
            return recursive_apply_pair(
                indices, self.tensor, _fetch_cpu, self.feature_shape, device, pin_memory)
        else:
            # CUDA to CUDA or CPU
            return recursive_apply_pair(
                indices, self.tensor, _fetch_cuda, device)
