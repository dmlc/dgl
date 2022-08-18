from .. import backend as F
from .base import FeatureStorage
from ..contrib.unified_tensor import UnifiedTensor
from ..contrib.gpu_cache import GPUCache

class GPUCachedTensorStorage(FeatureStorage):
    """FeatureStorage that slices features from a cached tensor and transfers it to a device."""
    def __init__(self, tensor, cache_size):
        flat_tensor = F.reshape(tensor, (tensor.shape[0], -1))
        self.storage = UnifiedTensor(flat_tensor, 'cuda')
        self.item_shape = tensor.shape[1:]
        self.cache = GPUCache(cache_size, self.storage.shape[1])

    def fetch(self, indices, device, pin_memory=False, **kwargs): # pylint: disable=unused-argument
        keys = indices.to('cuda')
        values, missing_index, missing_keys = self.cache.query(keys)
        missing_values = self.storage[missing_keys]
        values[missing_index] = missing_values
        self.cache.replace(missing_keys, missing_values)
        return F.copy_to(F.reshape(values, (values.shape[0],) + self.item_shape),
                        device, **kwargs)
