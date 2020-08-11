"""Define distributed tensor."""

import os

from .dist_context import is_initialized
from .kvstore import get_kvstore
from .. import utils
from .. import backend as F

def _get_data_name(name, part_policy):
    ''' This is to get the name of data in the kvstore.

    KVStore doesn't understand node data or edge data. We'll use a prefix to distinguish them.
    '''
    return part_policy + ':' + name

def _default_init_data(shape, dtype):
    return F.zeros(shape, dtype, F.cpu())

class DistTensor:
    ''' Distributed tensor.

    DistTensor references to a tensor stored in the distributed KVStore.
    When a DistTensor is created, it may reference to a tensor in the KVStore, or
    create a new one. The tensor is identified by the name passed to the constructor
    of DistTensor. If the name exists, DistTensor will reference the existing one.
    In this case, the shape and the data type should match the existing tensor.
    If the name doesn't exist, a new tensor will be created in the kvstore.

    If persistent=True when creating DistTesnor, the tensor in the KVStore will
    be persistent. Even if DistTensor is destroyed in the local trainer process,
    the tensor will still exist in KVStore. However, we do not allow an anonymous
    tensor to be persistent.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor
    dtype : dtype
        The dtype of the tensor
    name : string
        The name of the tensor.
    init_func : callable
        The function to initialize data in the tensor.
    part_policy : PartitionPolicy
        The partition policy of the tensor
    persistent : bool
        Whether the created tensor is persistent.
    '''
    def __init__(self, shape, dtype, name=None, init_func=None, part_policy=None,
                 persistent=False):
        self.kvstore = get_kvstore()
        self._shape = shape
        self._dtype = dtype

        part_policies = self.kvstore.all_possible_part_policy
        # If a user doesn't provide a partition policy, we should find one based on
        # the input shape.
        if part_policy is None:
            for policy_name in part_policies:
                policy = part_policies[policy_name]
                if policy.get_size() == shape[0]:
                    # If multiple partition policies match the input shape, we cannot
                    # decide which is the right one automatically. We should ask users
                    # to provide one.
                    assert part_policy is None, \
                            'Multiple partition policies match the input shape. ' \
                            + 'Please provide a partition policy explicitly.'
                    part_policy = policy
            assert part_policy is not None, \
                    'Cannot determine the partition policy. Please provide it.'

        self._part_policy = part_policy

        if init_func is None:
            init_func = _default_init_data
        exist_names = self.kvstore.data_name_list()
        # If a user doesn't provide a name, we generate a name ourselves.
        # We need to generate the name in a deterministic way.
        if name is None:
            assert not persistent, 'We cannot generate anonymous persistent distributed tensors'
            name = 'anonymous-' + str(len(exist_names) + 1)
        self._name = _get_data_name(name, part_policy.policy_str)
        self._persistent = persistent
        if self._name not in exist_names:
            self.kvstore.init_data(self._name, shape, dtype, part_policy, init_func)
            self._owner = True
        else:
            self._owner = False
            dtype1, shape1, _ = self.kvstore.get_data_meta(self._name)
            assert dtype == dtype1, 'The dtype does not match with the existing tensor'
            assert shape == shape1, 'The shape does not match with the existing tensor'

    def __del__(self):
        initialized = os.environ.get('DGL_DIST_MODE', 'standalone') == 'standalone' \
                or is_initialized()
        if not self._persistent and self._owner and initialized:
            self.kvstore.delete_data(self._name)

    def __getitem__(self, idx):
        idx = utils.toindex(idx)
        idx = idx.tousertensor()
        return self.kvstore.pull(name=self._name, id_tensor=idx)

    def __setitem__(self, idx, val):
        idx = utils.toindex(idx)
        idx = idx.tousertensor()
        # TODO(zhengda) how do we want to support broadcast (e.g., G.ndata['h'][idx] = 1).
        self.kvstore.push(name=self._name, id_tensor=idx, data_tensor=val)

    def __len__(self):
        return self._shape[0]

    @property
    def part_policy(self):
        ''' Return the partition policy '''
        return self._part_policy

    @property
    def shape(self):
        ''' Return the shape of the distributed tensor. '''
        return self._shape

    @property
    def dtype(self):
        ''' Return the data type of the distributed tensor. '''
        return self._dtype

    @property
    def name(self):
        ''' Return the name of the distributed tensor '''
        return self._name
