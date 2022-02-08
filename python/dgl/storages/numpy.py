"""Feature storage for ``numpy.memmap`` object."""
import numpy as np
from .base import FeatureStorage, ThreadedFuture, register_storage_wrapper
from .. import backend as F

@register_storage_wrapper(np.memmap)
class NumpyStorage(FeatureStorage):
    """FeatureStorage that asynchronously reads features from a ``numpy.memmap`` object."""
    def __init__(self, arr):
        self.arr = arr

    def _fetch(self, indices, device, pin_memory=False):    # pylint: disable=unused-argument
        result = F.zerocopy_from_numpy(self.arr[indices])
        result = F.copy_to(result, device)
        return result

    def fetch(self, indices, device, pin_memory=False):
        return ThreadedFuture(target=self._fetch, args=(indices, device, pin_memory))
