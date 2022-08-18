# /*!
#  *   Copyright (c) 2022, NVIDIA Corporation
#  *   All rights reserved.
#  *
#  *   Licensed under the Apache License, Version 2.0 (the "License");
#  *   you may not use this file except in compliance with the License.
#  *   You may obtain a copy of the License at
#  *
#  *       http://www.apache.org/licenses/LICENSE-2.0
#  *
#  *   Unless required by applicable law or agreed to in writing, software
#  *   distributed under the License is distributed on an "AS IS" BASIS,
#  *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  *   See the License for the specific language governing permissions and
#  *   limitations under the License.
#  *
#  * \file gpu_cache.py
#  * \brief API for managing a GPU Cache
#  */

from .. import backend as F
from .._ffi.function import _init_api
from ..storages import FeatureStorage
from .unified_tensor import UnifiedTensor

class GPUCache(object):
    """High-level wrapper for GPU embedding cache"""
    def __init__(self, num_items, num_feats, idtype=F.int64):
        assert idtype == F.int32 or idtype == F.int64
        self._cache = _CAPI_DGLGpuCacheCreate(num_items, num_feats, 32 if idtype == F.int32 else 64)
        self.idtype = idtype
        self.total_miss = 0
        self.total_queries = 0
    
    def query(self, keys):
        self.total_queries += keys.shape[0]
        keys = F.astype(keys, self.idtype)
        values, missing_index, missing_keys = _CAPI_DGLGpuCacheQuery(self._cache, F.to_dgl_nd(keys))
        self.total_miss += missing_keys.shape[0]
        return F.from_dgl_nd(values), F.from_dgl_nd(missing_index), F.from_dgl_nd(missing_keys)
    
    def replace(self, keys, values):
        keys = F.astype(keys, self.idtype)
        values = F.astype(values, F.float32)
        _CAPI_DGLGpuCacheReplace(self._cache, F.to_dgl_nd(keys), F.to_dgl_nd(values))
    
    @property
    def hit_rate(self):
        return 1 - self.total_miss / self.total_queries

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

_init_api("dgl.cuda", __name__)