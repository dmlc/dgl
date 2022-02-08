"""Feature storages for tensors across different frameworks."""
from .base import FeatureStorage
from .. import backend as F
from ..utils import recursive_apply_pair

def _fetch(indices, tensor, device):
    return F.copy_to(F.gather_row(tensor, indices), device)

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

    def fetch(self, indices, device, pin_memory=False):     # pylint: disable=unused-argument
        return recursive_apply_pair(indices, self.storage, _fetch, device)
