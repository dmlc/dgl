"""Feature storages for tensors across different frameworks."""
from .base import FeatureStorage
from .. import backend as F
from ..utils import recursive_apply_pair

def _fetch(indices, tensor, device):
    return F.copy_to(F.gather_row(tensor, indices), device)

class TensorStorage(FeatureStorage):
    """FeatureStorage that synchronously slices features from a tensor and transfers
    it to the given device.
    """
    def __init__(self, tensor):
        self.storage = tensor

    def fetch(self, indices, device, pin_memory=False):     # pylint: disable=unused-argument
        return recursive_apply_pair(indices, self.storage, _fetch, device)
