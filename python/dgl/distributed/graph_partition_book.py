"""Define graph partition book."""

import numpy as np

from .. import backend as F
from ..base import NID, EID
from .. import utils
from .shared_mem_utils import _to_shared_mem, _get_ndata_path, _get_edata_path, DTYPE_DICT
from .._ffi.ndarray import empty_shared_mem
from ..ndarray import exist_shared_mem_array

def _move_metadata_to_shared_mem(graph_name, num_nodes, num_edges, part_id,
                                 num_partitions, node_map, edge_map, is_range_part):
    ''' Move all metadata of the partition book to the shared memory.

    We need these metadata to construct graph partition book.
    '''
    meta = _to_shared_mem(F.tensor([int(is_range_part), num_nodes, num_edges,
                                    num_partitions, part_id]),
                          _get_ndata_path(graph_name, 'meta'))
    node_map = _to_shared_mem(node_map, _get_ndata_path(graph_name, 'node_map'))
    edge_map = _to_shared_mem(edge_map, _get_edata_path(graph_name, 'edge_map'))
    return meta, node_map, edge_map

def _get_shared_mem_metadata(graph_name):
    ''' Get the metadata of the graph through shared memory.

    The metadata includes the number of nodes and the number of edges. In the future,
    we can add more information, especially for heterograph.
    '''
    # The metadata has 5 elements: is_range_part, num_nodes, num_edges, num_partitions, part_id
    # We might need to extend the list in the future.
    shape = (5,)
    dtype = F.int64
    dtype = DTYPE_DICT[dtype]
    data = empty_shared_mem(_get_ndata_path(graph_name, 'meta'), False, shape, dtype)
    dlpack = data.to_dlpack()
    meta = F.asnumpy(F.zerocopy_from_dlpack(dlpack))
    is_range_part, num_nodes, num_edges, num_partitions, part_id = meta

    # Load node map
    length = num_partitions if is_range_part else num_nodes
    data = empty_shared_mem(_get_ndata_path(graph_name, 'node_map'), False, (length,), dtype)
    dlpack = data.to_dlpack()
    node_map = F.zerocopy_from_dlpack(dlpack)

    # Load edge_map
    length = num_partitions if is_range_part else num_edges
    data = empty_shared_mem(_get_edata_path(graph_name, 'edge_map'), False, (length,), dtype)
    dlpack = data.to_dlpack()
    edge_map = F.zerocopy_from_dlpack(dlpack)

    return is_range_part, part_id, num_partitions, node_map, edge_map


def get_shared_mem_partition_book(graph_name, graph_part):
    '''Get a graph partition book from shared memory.

    A graph partition book of a specific graph can be serialized to shared memory.
    We can reconstruct a graph partition book from shared memory.

    Parameters
    ----------
    graph_name : str
        The name of the graph.
    graph_part : DGLGraph
        The graph structure of a partition.

    Returns
    -------
    GraphPartitionBook
        A graph partition book for a particular partition.
    '''
    if not exist_shared_mem_array(_get_ndata_path(graph_name, 'meta')):
        return None
    is_range_part, part_id, num_parts, node_map, edge_map = _get_shared_mem_metadata(graph_name)
    if is_range_part == 1:
        return RangePartitionBook(part_id, num_parts, node_map, edge_map)
    else:
        return BasicPartitionBook(part_id, num_parts, node_map, edge_map, graph_part)

class GraphPartitionBook:
    """ The base class of the graph partition book.

    For distributed training, a graph is partitioned into multiple parts and is loaded
    in multiple machines. The partition book contains all necessary information to locate
    nodes and edges in the cluster.

    The partition book contains various partition information, including

    * the number of partitions,
    * the partition ID that a node or edge belongs to,
    * the node IDs and the edge IDs that a partition has.
    * the local IDs of nodes and edges in a partition.

    Currently, there are two classes that implement `GraphPartitionBook`:
    `BasicGraphPartitionBook` and `RangePartitionBook`. `BasicGraphPartitionBook`
    stores the mappings between every individual node/edge ID and partition ID on
    every machine, which usually consumes a lot of memory, while `RangePartitionBook`
    calculates the mapping between node/edge IDs and partition IDs based on some small
    metadata because nodes/edges have been relabeled to have IDs in the same partition
    fall in a contiguous ID range. `RangePartitionBook` is usually a preferred way to
    provide mappings between node/edge IDs and partition IDs.

    A graph partition book is constructed automatically when a graph is partitioned.
    When a graph partition is loaded, a graph partition book is loaded as well.
    Please see :py:meth:`~dgl.distributed.partition.partition_graph`,
    :py:meth:`~dgl.distributed.partition.load_partition` and
    :py:meth:`~dgl.distributed.partition.load_partition_book` for more details.
    """

    def shared_memory(self, graph_name):
        """Move the partition book to shared memory.

        Parameters
        ----------
        graph_name : str
            The graph name. This name will be used to read the partition book from shared
            memory in another process.
        """

    def num_partitions(self):
        """Return the number of partitions.

        Returns
        -------
        int
            number of partitions
        """

    def metadata(self):
        """Return the partition meta data.

        The meta data includes:

        * The machine ID.
        * Number of nodes and edges of each partition.

        Examples
        --------
        >>> print(g.get_partition_book().metadata())
        >>> [{'machine_id' : 0, 'num_nodes' : 3000, 'num_edges' : 5000},
        ...  {'machine_id' : 1, 'num_nodes' : 2000, 'num_edges' : 4888},
        ...  ...]

        Returns
        -------
        list[dict[str, any]]
            Meta data of each partition.
        """

    def nid2partid(self, nids):
        """From global node IDs to partition IDs

        Parameters
        ----------
        nids : tensor
            global node IDs

        Returns
        -------
        tensor
            partition IDs
        """

    def eid2partid(self, eids):
        """From global edge IDs to partition IDs

        Parameters
        ----------
        eids : tensor
            global edge IDs

        Returns
        -------
        tensor
            partition IDs
        """

    def partid2nids(self, partid):
        """From partition id to global node IDs

        Parameters
        ----------
        partid : int
            partition id

        Returns
        -------
        tensor
            node IDs
        """

    def partid2eids(self, partid):
        """From partition id to global edge IDs

        Parameters
        ----------
        partid : int
            partition id

        Returns
        -------
        tensor
            edge IDs
        """

    def nid2localnid(self, nids, partid):
        """Get local node IDs within the given partition.

        Parameters
        ----------
        nids : tensor
            global node IDs
        partid : int
            partition ID

        Returns
        -------
        tensor
             local node IDs
        """

    def eid2localeid(self, eids, partid):
        """Get the local edge ids within the given partition.

        Parameters
        ----------
        eids : tensor
            global edge ids
        partid : int
            partition ID

        Returns
        -------
        tensor
             local edge ids
        """

    @property
    def partid(self):
        """Get the current partition id

        Return
        ------
        int
            The partition id of current machine
        """

class BasicPartitionBook(GraphPartitionBook):
    """This provides the most flexible way to store parition information.

    The partition book maintains the mapping of every single node IDs and edge IDs to
    partition IDs. This is very flexible at the coast of large memory consumption.
    On a large graph, the mapping consumes significant memory and this partition book
    is not recommended.

    Parameters
    ----------
    part_id : int
        partition id of current partition book
    num_parts : int
        number of total partitions
    node_map : tensor
        global node id mapping to partition id
    edge_map : tensor
        global edge id mapping to partition id
    part_graph : DGLGraph
        The graph partition structure.
    """
    def __init__(self, part_id, num_parts, node_map, edge_map, part_graph):
        assert part_id >= 0, 'part_id cannot be a negative number.'
        assert num_parts > 0, 'num_parts must be greater than zero.'
        self._part_id = int(part_id)
        self._num_partitions = int(num_parts)
        self._nid2partid = F.tensor(node_map)
        assert F.dtype(self._nid2partid) == F.int64, \
                'the node map must be stored in an integer array'
        self._eid2partid = F.tensor(edge_map)
        assert F.dtype(self._eid2partid) == F.int64, \
                'the edge map must be stored in an integer array'
        # Get meta data of the partition book.
        self._partition_meta_data = []
        _, nid_count = np.unique(F.asnumpy(self._nid2partid), return_counts=True)
        _, eid_count = np.unique(F.asnumpy(self._eid2partid), return_counts=True)
        for partid in range(self._num_partitions):
            part_info = {}
            part_info['machine_id'] = partid
            part_info['num_nodes'] = int(nid_count[partid])
            part_info['num_edges'] = int(eid_count[partid])
            self._partition_meta_data.append(part_info)
        # Get partid2nids
        self._partid2nids = []
        sorted_nid = F.tensor(np.argsort(F.asnumpy(self._nid2partid)))
        start = 0
        for offset in nid_count:
            part_nids = sorted_nid[start:start+offset]
            start += offset
            self._partid2nids.append(part_nids)
        # Get partid2eids
        self._partid2eids = []
        sorted_eid = F.tensor(np.argsort(F.asnumpy(self._eid2partid)))
        start = 0
        for offset in eid_count:
            part_eids = sorted_eid[start:start+offset]
            start += offset
            self._partid2eids.append(part_eids)
        # Get nidg2l
        self._nidg2l = [None] * self._num_partitions
        global_id = part_graph.ndata[NID]
        max_global_id = np.amax(F.asnumpy(global_id))
        # TODO(chao): support int32 index
        g2l = F.zeros((max_global_id+1), F.int64, F.context(global_id))
        g2l = F.scatter_row(g2l, global_id, F.arange(0, len(global_id)))
        self._nidg2l[self._part_id] = g2l
        # Get eidg2l
        self._eidg2l = [None] * self._num_partitions
        global_id = part_graph.edata[EID]
        max_global_id = np.amax(F.asnumpy(global_id))
        # TODO(chao): support int32 index
        g2l = F.zeros((max_global_id+1), F.int64, F.context(global_id))
        g2l = F.scatter_row(g2l, global_id, F.arange(0, len(global_id)))
        self._eidg2l[self._part_id] = g2l
        # node size and edge size
        self._edge_size = len(self.partid2eids(self._part_id))
        self._node_size = len(self.partid2nids(self._part_id))

    def shared_memory(self, graph_name):
        """Move data to shared memory.
        """
        self._meta, self._nid2partid, self._eid2partid = _move_metadata_to_shared_mem(
            graph_name, self._num_nodes(), self._num_edges(), self._part_id, self._num_partitions,
            self._nid2partid, self._eid2partid, False)

    def num_partitions(self):
        """Return the number of partitions.
        """
        return self._num_partitions

    def metadata(self):
        """Return the partition meta data.
        """
        return self._partition_meta_data

    def _num_nodes(self):
        """ The total number of nodes
        """
        return len(self._nid2partid)

    def _num_edges(self):
        """ The total number of edges
        """
        return len(self._eid2partid)

    def nid2partid(self, nids):
        """From global node IDs to partition IDs
        """
        return F.gather_row(self._nid2partid, nids)

    def eid2partid(self, eids):
        """From global edge IDs to partition IDs
        """
        return F.gather_row(self._eid2partid, eids)

    def partid2nids(self, partid):
        """From partition id to global node IDs
        """
        return self._partid2nids[partid]

    def partid2eids(self, partid):
        """From partition id to global edge IDs
        """
        return self._partid2eids[partid]

    def nid2localnid(self, nids, partid):
        """Get local node IDs within the given partition.
        """
        if partid != self._part_id:
            raise RuntimeError('Now GraphPartitionBook does not support \
                getting remote tensor of nid2localnid.')
        return F.gather_row(self._nidg2l[partid], nids)

    def eid2localeid(self, eids, partid):
        """Get the local edge ids within the given partition.
        """
        if partid != self._part_id:
            raise RuntimeError('Now GraphPartitionBook does not support \
                getting remote tensor of eid2localeid.')
        return F.gather_row(self._eidg2l[partid], eids)

    @property
    def partid(self):
        """Get the current partition id
        """
        return self._part_id


class RangePartitionBook(GraphPartitionBook):
    """This partition book supports more efficient storage of partition information.

    This partition book is used if the nodes and edges of a graph partition are assigned
    with contiguous IDs. It uses very small amount of memory to store the partition
    information.

    Parameters
    ----------
    part_id : int
        partition id of current partition book
    num_parts : int
        number of total partitions
    node_map : tensor
        map global node id to partition id
    edge_map : tensor
        map global edge id to partition id
    """
    def __init__(self, part_id, num_parts, node_map, edge_map):
        assert part_id >= 0, 'part_id cannot be a negative number.'
        assert num_parts > 0, 'num_parts must be greater than zero.'
        self._partid = part_id
        self._num_partitions = num_parts
        if not isinstance(node_map, np.ndarray):
            node_map = F.asnumpy(node_map)
        if not isinstance(edge_map, np.ndarray):
            edge_map = F.asnumpy(edge_map)
        self._node_map = node_map
        self._edge_map = edge_map
        # Get meta data of the partition book
        self._partition_meta_data = []
        for partid in range(self._num_partitions):
            nrange_start = node_map[partid - 1] if partid > 0 else 0
            nrange_end = node_map[partid]
            erange_start = edge_map[partid - 1] if partid > 0 else 0
            erange_end = edge_map[partid]
            part_info = {}
            part_info['machine_id'] = partid
            part_info['num_nodes'] = int(nrange_end - nrange_start)
            part_info['num_edges'] = int(erange_end - erange_start)
            self._partition_meta_data.append(part_info)

    def shared_memory(self, graph_name):
        """Move data to shared memory.
        """
        self._meta = _move_metadata_to_shared_mem(
            graph_name, self._num_nodes(), self._num_edges(), self._partid,
            self._num_partitions, F.tensor(self._node_map), F.tensor(self._edge_map), True)

    def num_partitions(self):
        """Return the number of partitions.
        """
        return self._num_partitions


    def _num_nodes(self):
        """ The total number of nodes
        """
        return int(self._node_map[-1])

    def _num_edges(self):
        """ The total number of edges
        """
        return int(self._edge_map[-1])

    def metadata(self):
        """Return the partition meta data.
        """
        return self._partition_meta_data


    def nid2partid(self, nids):
        """From global node IDs to partition IDs
        """
        nids = utils.toindex(nids)
        ret = np.searchsorted(self._node_map, nids.tonumpy(), side='right')
        ret = utils.toindex(ret)
        return ret.tousertensor()


    def eid2partid(self, eids):
        """From global edge IDs to partition IDs
        """
        eids = utils.toindex(eids)
        ret = np.searchsorted(self._edge_map, eids.tonumpy(), side='right')
        ret = utils.toindex(ret)
        return ret.tousertensor()


    def partid2nids(self, partid):
        """From partition id to global node IDs
        """
        # TODO do we need to cache it?
        start = self._node_map[partid - 1] if partid > 0 else 0
        end = self._node_map[partid]
        return F.arange(start, end)


    def partid2eids(self, partid):
        """From partition id to global edge IDs
        """
        # TODO do we need to cache it?
        start = self._edge_map[partid - 1] if partid > 0 else 0
        end = self._edge_map[partid]
        return F.arange(start, end)


    def nid2localnid(self, nids, partid):
        """Get local node IDs within the given partition.
        """
        if partid != self._partid:
            raise RuntimeError('Now RangePartitionBook does not support \
                getting remote tensor of nid2localnid.')

        nids = utils.toindex(nids)
        nids = nids.tousertensor()
        start = self._node_map[partid - 1] if partid > 0 else 0
        return nids - int(start)


    def eid2localeid(self, eids, partid):
        """Get the local edge ids within the given partition.
        """
        if partid != self._partid:
            raise RuntimeError('Now RangePartitionBook does not support \
                getting remote tensor of eid2localeid.')

        eids = utils.toindex(eids)
        eids = eids.tousertensor()
        start = self._edge_map[partid - 1] if partid > 0 else 0
        return eids - int(start)


    @property
    def partid(self):
        """Get the current partition id
        """
        return self._partid

NODE_PART_POLICY = 'node'
EDGE_PART_POLICY = 'edge'

class PartitionPolicy(object):
    """This defines a partition policy for a distributed tensor or distributed embedding.

    When DGL shards tensors and stores them in a cluster of machines, it requires
    partition policies that map rows of the tensors to machines in the cluster.

    Although an arbitrary partition policy can be defined, DGL currently supports
    two partition policies for mapping nodes and edges to machines. To define a partition
    policy from a graph partition book, users need to specify the policy name ('node' or 'edge').

    Parameters
    ----------
    policy_str : str
        Partition policy name, e.g., 'edge' or 'node'.
    partition_book : GraphPartitionBook
        A graph partition book
    """
    def __init__(self, policy_str, partition_book):
        # TODO(chao): support more policies for HeteroGraph
        assert policy_str in (EDGE_PART_POLICY, NODE_PART_POLICY), \
                'policy_str must be \'edge\' or \'node\'.'
        self._policy_str = policy_str
        self._part_id = partition_book.partid
        self._partition_book = partition_book

    @property
    def policy_str(self):
        """Get the policy name

        Returns
        -------
        str
            The name of the partition policy.
        """
        return self._policy_str

    @property
    def part_id(self):
        """Get partition ID

        Returns
        -------
        int
            The partition ID
        """
        return self._part_id

    @property
    def partition_book(self):
        """Get partition book

        Returns
        -------
        GraphPartitionBook
            The graph partition book
        """
        return self._partition_book

    def to_local(self, id_tensor):
        """Mapping global ID to local ID.

        Parameters
        ----------
        id_tensor : tensor
            Gloabl ID tensor

        Return
        ------
        tensor
            local ID tensor
        """
        if self._policy_str == EDGE_PART_POLICY:
            return self._partition_book.eid2localeid(id_tensor, self._part_id)
        elif self._policy_str == NODE_PART_POLICY:
            return self._partition_book.nid2localnid(id_tensor, self._part_id)
        else:
            raise RuntimeError('Cannot support policy: %s ' % self._policy_str)

    def to_partid(self, id_tensor):
        """Mapping global ID to partition ID.

        Parameters
        ----------
        id_tensor : tensor
            Global ID tensor

        Return
        ------
        tensor
            partition ID
        """
        if self._policy_str == EDGE_PART_POLICY:
            return self._partition_book.eid2partid(id_tensor)
        elif self._policy_str == NODE_PART_POLICY:
            return self._partition_book.nid2partid(id_tensor)
        else:
            raise RuntimeError('Cannot support policy: %s ' % self._policy_str)

    def get_part_size(self):
        """Get data size of current partition.

        Returns
        -------
        int
            data size
        """
        if self._policy_str == EDGE_PART_POLICY:
            return len(self._partition_book.partid2eids(self._part_id))
        elif self._policy_str == NODE_PART_POLICY:
            return len(self._partition_book.partid2nids(self._part_id))
        else:
            raise RuntimeError('Cannot support policy: %s ' % self._policy_str)

    def get_size(self):
        """Get the full size of the data.

        Returns
        -------
        int
            data size
        """
        if self._policy_str == EDGE_PART_POLICY:
            return self._partition_book._num_edges()
        elif self._policy_str == NODE_PART_POLICY:
            return self._partition_book._num_nodes()
        else:
            raise RuntimeError('Cannot support policy: %s ' % self._policy_str)
