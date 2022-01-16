import torch
from .base import FeatureStorage

class TensorStorage(FeatureStorage):
    def __init__(self, tensor):
        self.tensor = tensor
        self.feature_shape = tensor.shape[1:]
        self.is_cuda = (tensor.device.type == 'cuda')

    def fetch(self, indices, device, pin_memory=False):
        device = torch.device(device)
        if not self.is_cuda:
            # CPU to CPU or CUDA - use pin_memory and async transfer if possible
            result = torch.empty(
                indices.shape[0], *self.feature_shape, dtype=self.tensor.dtype,
                pin_memory=pin_memory)
            torch.index_select(self.tensor, 0, indices, out=result)
            result = result.to(device, non_blocking=True)
        else:
            # CUDA to CUDA or CPU
            result = torch.index_select(self.tensor, 0, indices).to(device)
        return result
