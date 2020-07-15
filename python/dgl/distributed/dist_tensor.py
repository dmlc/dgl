"""Define distributed tensor."""

from .graph_partition_book import PartitionPolicy, NODE_PART_POLICY, EDGE_PART_POLICY
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

    This is a wrapper to access a tensor stored in the distributed KVStore.
    This wrapper provides an interface similar to the local tensor.

    If a user tries to create a new tensor with a name that exists in the KVStore,
    the creation will fail by default. However, if reuse_if_exist=True, it tries
    to reuse the existing tensor in the KVStore if the shape and dtype match.

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
    part_policy : PartitionPolicy
        The partition policy of the tensor
    init_func : callable
        The function to initialize data in the tensor.
    create_new : bool
        Whether or not to create a new tensor in the KVStore.
    reuse_if_exist : bool
        Reuse the existing tensor if create_new=True.
    '''
    def __init__(self, g, shape, dtype, name, part_policy=None, init_func=None,
                 create_new=True, reuse_if_exist=False):
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
        self._name = _get_data_name(name, part_policy.policy_str)
        if create_new:
            if reuse_if_exist and self._name in g._client.data_name_list():
                dtype1, shape1, _ = g._client.get_data_meta(self._name)
                assert dtype == dtype1, 'The dtype does not match with the existing tensor'
                assert shape == shape1, 'The shape does not match with the existing tensor'
            else:
                g._client.init_data(self._name, shape, dtype, part_policy, init_func)
        else:
            dtype1, shape1, _ = g._client.get_data_meta(self._name)
            assert dtype == dtype1, 'The dtype does not match with the existing tensor'
            assert shape == shape1, 'The shape does not match with the existing tensor'

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
