"""Feature storages for tensors across different frameworks."""
from .base import FeatureStorage
from .. import backend as F

class BaseTensorStorage(FeatureStorage):
    """FeatureStorage that synchronously slices features from a tensor and transfers
    it to the given device.
    """
    def __init__(self, tensor):
        self.storage = tensor

    def fetch(self, indices, device, pin_memory=False, **kwargs): # pylint: disable=unused-argument
        return F.copy_to(F.gather_row(tensor, indices), device, **kwargs)
