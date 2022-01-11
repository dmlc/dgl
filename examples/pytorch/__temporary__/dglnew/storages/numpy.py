import numpy
import torch
from .base import FeatureStorage

class NumpyStorage(FeatureStorage):
    def __init__(self, arr):
        self.arr = arr
        self.feature_shape = arr.shape[1:]

    def fetch(self, indices, device, pin_memory=False):
        device = torch.device(device)
        result = torch.from_numpy(self.arr[indices])
        if pin_memory:
            result = result.pin_memory()
        result = result.to(device, non_blocking=True)
        return result
