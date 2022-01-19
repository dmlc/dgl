import numpy
from .base import FeatureStorage
from .. import backend as F

class NumpyStorage(FeatureStorage):
    def __init__(self, arr):
        self.arr = arr

    def fetch(self, indices, device, pin_memory=False):
        result = F.zerocopy_from_numpy(self.arr[indices])
        result = F.copy_to(result, device)
        return result
