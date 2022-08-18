from .. import backend as F
from ..storages import GPUCachedTensorStorage

class GPUCachedTensor: #GPUCachedTensor
    def __init__(self, input, cache_size):
        self.storage = GPUCachedTensorStorage(input, cache_size)

    def __len__(self):
        return len(self.storage.storage)

    def __repr__(self):
        return self.storage.storage.__repr__()

    def __getitem__(self, key):
        '''Perform zero-copy access from GPU if the context of
        the key is cuda. Otherwise, just safely fallback to the
        backend specific indexing scheme.

        Parameters
        ----------
        key : Tensor
            Tensor which contains the index ids
        '''
        return self.storage.fetch(key, F.context(key))

    def __setitem__(self, key, val):
        self.storage.storage[key] = val
        self.storage.cache.replace(key, val)

    @property
    def shape(self):
        """Shape of this tensor"""
        return self.storage.storage.shape

    @property
    def dtype(self):
        """Type of this tensor"""
        return self.storage.storage.dtype

    @property
    def device(self):
        """Device of this tensor"""
        return self.storage.storage.device
    
    @property
    def hit_rate(self):
        return self.storage.cache.hit_rate
