"""Define distributed tensor."""

import os
import uuid

from .graph_partition_book import PartitionPolicy, NODE_PART_POLICY, EDGE_PART_POLICY
from .rpc_client import is_initialized
from ..base import DGLError
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
    g : DistGraph
        The distributed graph object.
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
    def __init__(self, g, shape, dtype, name=None, init_func=None, part_policy=None,
                 persistent=False):
        self.kvstore = g._client
        self._shape = shape
        self._dtype = dtype

        if part_policy is None:
            assert shape[0] != g.number_of_nodes() or shape[0] != g.number_of_edges(), \
                    'Cannot determine the partition policy. Please provide it.'
            if shape[0] == g.number_of_nodes():
                part_policy = PartitionPolicy(NODE_PART_POLICY, g.get_partition_book())
            elif shape[0] == g.number_of_edges():
                part_policy = PartitionPolicy(EDGE_PART_POLICY, g.get_partition_book())
            else:
                raise DGLError('Cannot determine the partition policy. Please provide it.')

        self._part_policy = part_policy

        if init_func is None:
            init_func = _default_init_data
        # If a user doesn't provide a name, we generate a name ourselves.
        if name is None:
            assert not persistent, 'We cannot generate anonymous persistent distributed tensors'
            name = uuid.uuid4().hex[:10]
        self._name = _get_data_name(name, part_policy.policy_str)
        self._persistent = persistent
        if self._name not in g._client.data_name_list():
            g._client.init_data(self._name, shape, dtype, part_policy, init_func)
            self._owner = True
        else:
            self._owner = False
            dtype1, shape1, _ = g._client.get_data_meta(self._name)
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
