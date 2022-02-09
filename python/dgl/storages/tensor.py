"""Feature storages for tensors across different frameworks."""
from functools import lru_cache
from .base import FeatureStorage
from .. import backend as F
from ..utils import recursive_apply_pair
from ..contrib.unified_tensor import UnifiedTensor

class BaseTensorStorage(FeatureStorage):
    """FeatureStorage that synchronously slices features from a tensor and transfers
    it to the given device.
    """
    def __init__(self, tensor):
        self.storage = tensor   # also sets _feature_shape and _is_cuda

    @property
    def storage(self):
        return self._storage

    # Because DGL's Frame would change self.storage, we need to make self.storage a property
    # so that feature shape and is_cuda flag gets updated at the same time.
    @storage.setter
    def storage(self, val):
        self._storage = val
        if val is not None:
            self._feature_shape = val.shape[1:]
            self._is_cuda = (val.device.type == 'cuda')
        self.get_unified_tensor.cache_clear()

    @lru_cache(maxsize=None)
    def get_unified_tensor(self, device):
        return UnifiedTensor(self.storage, device)

    def fetch(self, indices, device, pin_memory=False):     # pylint: disable=unused-argument
        return F.copy_to(F.gather_row(tensor, indices), device)
