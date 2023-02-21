"""Feature storage for ``numpy.memmap`` object."""
import numpy as np

from .. import backend as F
from .base import FeatureStorage, register_storage_wrapper, ThreadedFuture


@register_storage_wrapper(np.memmap)
class NumpyStorage(FeatureStorage):
    """FeatureStorage that asynchronously reads features from a ``numpy.memmap`` object."""

    def __init__(self, arr):
        self.arr = arr

    # pylint: disable=unused-argument
    def _fetch(self, indices, device, pin_memory=False):
        result = F.zerocopy_from_numpy(self.arr[indices])
        result = F.copy_to(result, device)
        return result

    # pylint: disable=unused-argument
    def fetch(self, indices, device, pin_memory=False, **kwargs):
        return ThreadedFuture(
            target=self._fetch, args=(indices, device, pin_memory)
        )
