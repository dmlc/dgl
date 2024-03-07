"""Define graph partition book."""

import pickle
from abc import ABC

import numpy as np

from .. import backend as F, utils
from .._ffi.ndarray import empty_shared_mem
from ..base import DGLError
from ..ndarray import exist_shared_mem_array
from ..partition import NDArrayPartition
from .constants import DEFAULT_ETYPE, DEFAULT_NTYPE
from .id_map import IdMap
from .shared_mem_utils import (
    _get_edata_path,
    _get_ndata_path,
    _to_shared_mem,
    DTYPE_DICT,
)

CANONICAL_ETYPE_DELIMITER = ":"


def _etype_tuple_to_str(c_etype):
    """Convert canonical etype from tuple to string.

    Examples
    --------
    >>> c_etype = ('user', 'like', 'item')
    >>> c_etype_str = _etype_tuple_to_str(c_etype)
    >>> print(c_etype_str)
    'user:like:item'

    """
    assert isinstance(c_etype, tuple) and len(c_etype) == 3, (
        "Passed-in canonical etype should be in format of (str, str, str). "
        f"But got {c_etype}."
    )
    return CANONICAL_ETYPE_DELIMITER.join(c_etype)


def _etype_str_to_tuple(c_etype):
    """Convert canonical etype from tuple to string.

    Examples
    --------
    >>> c_etype_str = 'user:like:item'
    >>> c_etype = _etype_str_to_tuple(c_etype_str)
    >>> print(c_etype)
    ('user', 'like', 'item')

    """
    ret = tuple(c_etype.split(CANONICAL_ETYPE_DELIMITER))
    assert len(ret) == 3, (
        "Passed-in canonical etype should be in format of 'str:str:str'. "
        f"But got {c_etype}."
    )
    return ret


def _move_metadata_to_shared_mem(
    graph_name,
    num_nodes,
    num_edges,
    part_id,
    num_partitions,
    node_map,
    edge_map,
    is_range_part,
):
    """Move all metadata of the partition book to the shared memory.

    These metadata will be used to construct graph partition book.

    Parameters
    ----------
    graph_name : str
        The name of the graph
    num_nodes : int
        The total number of nodes
    num_edges : int
        The total number of edges
    part_id : int
        The partition ID.
    num_partitions : int
        The number of physical partitions generated for the graph.
    node_map : Tensor
        It stores the mapping information from node IDs to partitions. With range partitioning,
        the tensor stores the serialized result of partition ranges.
    edge_map : Tensor
        It stores the mapping information from edge IDs to partitions. With range partitioning,
        the tensor stores the serialized result of partition ranges.
    is_range_part : bool
        Indicate that we use a range partition. This is important for us to deserialize data
        in node_map and edge_map.

    Returns
    -------
    (Tensor, Tensor, Tensor)
        The first tensor stores the serialized metadata, the second tensor stores the serialized
        node map and the third tensor stores the serialized edge map. All tensors are stored in
        shared memory.
    """
    meta = _to_shared_mem(
        F.tensor(
            [
                int(is_range_part),
                num_nodes,
                num_edges,
                num_partitions,
                part_id,
                len(node_map),
                len(edge_map),
            ]
        ),
        _get_ndata_path(graph_name, "meta"),
    )
    node_map = _to_shared_mem(node_map, _get_ndata_path(graph_name, "node_map"))
    edge_map = _to_shared_mem(edge_map, _get_edata_path(graph_name, "edge_map"))
    return meta, node_map, edge_map


def _get_shared_mem_metadata(graph_name):
    """Get the metadata of the graph from shared memory.

    The server serializes the metadata of a graph and store them in shared memory.
    The client needs to deserialize the data in shared memory and get the metadata
    of the graph.

    Parameters
    ----------
    graph_name : str
        The name of the graph. We can use the graph name to find the shared memory name.

    Returns
    -------
    (bool, int, int, Tensor, Tensor)
        The first element indicates whether it is range partitioning;
        the second element is the partition ID;
        the third element is the number of partitions;
        the fourth element is the tensor that stores the serialized result of node maps;
        the fifth element is the tensor that stores the serialized result of edge maps.
    """
    # The metadata has 7 elements: is_range_part, num_nodes, num_edges, num_partitions, part_id,
    # the length of node map and the length of the edge map.
    shape = (7,)
    dtype = F.int64
    dtype = DTYPE_DICT[dtype]
    data = empty_shared_mem(
        _get_ndata_path(graph_name, "meta"), False, shape, dtype
    )
    dlpack = data.to_dlpack()
    meta = F.asnumpy(F.zerocopy_from_dlpack(dlpack))
    (
        is_range_part,
        _,
        _,
        num_partitions,
        part_id,
        node_map_len,
        edge_map_len,
    ) = meta

    # Load node map
    data = empty_shared_mem(
        _get_ndata_path(graph_name, "node_map"), False, (node_map_len,), dtype
    )
    dlpack = data.to_dlpack()
    node_map = F.zerocopy_from_dlpack(dlpack)

    # Load edge_map
    data = empty_shared_mem(
        _get_edata_path(graph_name, "edge_map"), False, (edge_map_len,), dtype
    )
    dlpack = data.to_dlpack()
    edge_map = F.zerocopy_from_dlpack(dlpack)

    return is_range_part, part_id, num_partitions, node_map, edge_map


def get_shared_mem_partition_book(graph_name):
    """Get a graph partition book from shared memory.

    A graph partition book of a specific graph can be serialized to shared memory.
    We can reconstruct a graph partition book from shared memory.

    Parameters
    ----------
    graph_name : str
        The name of the graph.

    Returns
    -------
    GraphPartitionBook
        A graph partition book for a particular partition.
    """
    if not exist_shared_mem_array(_get_ndata_path(graph_name, "meta")):
        return None
    (
        is_range_part,
        part_id,
        num_parts,
        node_map_data,
        edge_map_data,
    ) = _get_shared_mem_metadata(graph_name)
    if is_range_part == 1:
        # node ID ranges and edge ID ranges are stored in the order of node type IDs
        # and edge type IDs.
        node_map = {}
        ntypes = {}
        # node_map_data and edge_map_data were serialized with pickle and converted into
        # a list of bytes and then stored in a numpy array before being placed in shared
        # memory. To deserialize, we need to reverse the process.
        node_map_data = pickle.loads(bytes(F.asnumpy(node_map_data).tolist()))
        for i, (ntype, nid_range) in enumerate(node_map_data):
            ntypes[ntype] = i
            node_map[ntype] = nid_range

        edge_map = {}
        etypes = {}
        edge_map_data = pickle.loads(bytes(F.asnumpy(edge_map_data).tolist()))
        for i, (etype, eid_range) in enumerate(edge_map_data):
            etypes[etype] = i
            edge_map[etype] = eid_range
        return RangePartitionBook(
            part_id, num_parts, node_map, edge_map, ntypes, etypes
        )
    else:
        raise TypeError("Only RangePartitionBook is supported currently.")


def get_node_partition_from_book(book, device):
    """Get an NDArrayPartition of the nodes from a RangePartitionBook.

    Parameters
    ----------
    book : RangePartitionBook
        The partition book to extract the node partition from.
    device : Device context object.
        The location to node partition is to be used.

    Returns
    -------
    NDarrayPartition
        The NDArrayPartition object for the nodes in the graph.
    """
    assert isinstance(book, RangePartitionBook), (
        "Can only convert " "RangePartitionBook to NDArrayPartition."
    )
    # create prefix-sum array on host
    max_node_ids = F.zerocopy_from_numpy(book._max_node_ids)
    cpu_range = F.cat(
        [F.tensor([0], dtype=F.dtype(max_node_ids)), max_node_ids + 1], dim=0
    )
    gpu_range = F.copy_to(cpu_range, ctx=device)

    # convert from numpy
    array_size = int(F.as_scalar(cpu_range[-1]))
    num_parts = book.num_partitions()

    return NDArrayPartition(
        array_size, num_parts, mode="range", part_ranges=gpu_range
    )


class GraphPartitionBook(ABC):
    """The base class of the graph partition book.

    For distributed training, a graph is partitioned into multiple parts and is loaded
    in multiple machines. The partition book contains all necessary information to locate
    nodes and edges in the cluster.

    The partition book contains various partition information, including

    * the number of partitions,
    * the partition ID that a node or edge belongs to,
    * the node IDs and the edge IDs that a partition has.
    * the local IDs of nodes and edges in a partition.

    Currently, only one class that implement ``GraphPartitionBook``
    :``RangePartitionBook``. It calculates the mapping between node/edge IDs
    and partition IDs based on some small metadata because nodes/edges have been
    relabeled to have IDs in the same partition fall in a contiguous ID range.

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

    def nid2partid(self, nids, ntype):
        """From global node IDs to partition IDs

        Parameters
        ----------
        nids : tensor
            global node IDs
        ntype : str
            The node type

        Returns
        -------
        tensor
            partition IDs
        """

    def eid2partid(self, eids, etype):
        """From global edge IDs to partition IDs

        Parameters
        ----------
        eids : tensor
            global edge IDs
        etype : str or (str, str, str)
            The edge type

        Returns
        -------
        tensor
            partition IDs
        """

    def partid2nids(self, partid, ntype):
        """From partition id to global node IDs

        Parameters
        ----------
        partid : int
            partition id
        ntype : str
            The node type

        Returns
        -------
        tensor
            node IDs
        """

    def partid2eids(self, partid, etype):
        """From partition id to global edge IDs

        Parameters
        ----------
        partid : int
            partition id
        etype : str or (str, str, str)
            The edge type

        Returns
        -------
        tensor
            edge IDs
        """

    def nid2localnid(self, nids, partid, ntype):
        """Get local node IDs within the given partition.

        Parameters
        ----------
        nids : tensor
            global node IDs
        partid : int
            partition ID
        ntype : str
            The node type

        Returns
        -------
        tensor
             local node IDs
        """

    def eid2localeid(self, eids, partid, etype):
        """Get the local edge ids within the given partition.

        Parameters
        ----------
        eids : tensor
            global edge IDs
        partid : int
            partition ID
        etype : str or (str, str, str)
            The edge type

        Returns
        -------
        tensor
             local edge IDs
        """

    @property
    def partid(self):
        """Get the current partition ID

        Return
        ------
        int
            The partition ID of current machine
        """

    @property
    def ntypes(self):
        """Get the list of node types"""

    @property
    def etypes(self):
        """Get the list of edge types"""

    @property
    def canonical_etypes(self):
        """Get the list of canonical edge types

        Returns
        -------
        list[(str, str, str)]
            A list of canonical etypes
        """

    def to_canonical_etype(self, etype):
        """Convert an edge type to the corresponding canonical edge type.

        Parameters
        ----------
        etype : str or (str, str, str)
            The edge type

        Returns
        -------
        (str, str, str)
            The corresponding canonical edge type
        """

    @property
    def is_homogeneous(self):
        """check if homogeneous"""
        return not (len(self.etypes) > 1 or len(self.ntypes) > 1)

    def map_to_per_ntype(self, ids):
        """Map homogeneous node IDs to type-wise IDs and node types.

        Parameters
        ----------
        ids : tensor
            Homogeneous node IDs.

        Returns
        -------
        (tensor, tensor)
            node type IDs and type-wise node IDs.
        """

    def map_to_per_etype(self, ids):
        """Map homogeneous edge IDs to type-wise IDs and edge types.

        Parameters
        ----------
        ids : tensor
            Homogeneous edge IDs.

        Returns
        -------
        (tensor, tensor)
            edge type IDs and type-wise edge IDs.
        """

    def map_to_homo_nid(self, ids, ntype):
        """Map type-wise node IDs and type IDs to homogeneous node IDs.

        Parameters
        ----------
        ids : tensor
            Type-wise node Ids
        ntype : str
            node type

        Returns
        -------
        Tensor
            Homogeneous node IDs.
        """

    def map_to_homo_eid(self, ids, etype):
        """Map type-wise edge IDs and type IDs to homogeneous edge IDs.

        Parameters
        ----------
        ids : tensor
            Type-wise edge Ids
        etype : str or (str, str, str)
            The edge type

        Returns
        -------
        Tensor
            Homogeneous edge IDs.
        """


class RangePartitionBook(GraphPartitionBook):
    """This partition book supports more efficient storage of partition information.

    This partition book is used if the nodes and edges of a graph partition are assigned
    with contiguous IDs. It uses very small amount of memory to store the partition
    information.

    Parameters
    ----------
    part_id : int
        partition ID of current partition book
    num_parts : int
        number of total partitions
    node_map : dict[str, Tensor]
        Global node ID ranges within partitions for each node type. The key is the node type
        name in string. The value is a tensor of shape :math:`(K, 2)`, where :math:`K` is
        the number of partitions. Each row has two integers: the starting and the ending IDs
        for a particular node type in a partition. For example, all nodes of type ``"T"`` in
        partition ``i`` has ID range ``node_map["T"][i][0]`` to ``node_map["T"][i][1]``.
    edge_map : dict[(str, str, str), Tensor]
        Global edge ID ranges within partitions for each edge type. The key is the edge type
        name in string. The value is a tensor of shape :math:`(K, 2)`, where :math:`K` is
        the number of partitions. Each row has two integers: the starting and the ending IDs
        for a particular edge type in a partition. For example, all edges of type ``"T"`` in
        partition ``i`` has ID range ``edge_map["T"][i][0]`` to ``edge_map["T"][i][1]``.
    ntypes : dict[str, int]
        map ntype strings to ntype IDs.
    etypes : dict[(str, str, str), int]
        map canonical etypes to etype IDs.

    """

    def __init__(self, part_id, num_parts, node_map, edge_map, ntypes, etypes):
        assert part_id >= 0, "part_id cannot be a negative number."
        assert num_parts > 0, "num_parts must be greater than zero."
        self._partid = part_id
        self._num_partitions = num_parts
        self._ntypes = [None] * len(ntypes)
        self._etypes = [None] * len(etypes)
        self._canonical_etypes = [None] * len(etypes)
        # map etypes to canonical ones
        self._etype2canonical = {}
        for ntype in ntypes:
            ntype_id = ntypes[ntype]
            self._ntypes[ntype_id] = ntype
        assert all(
            ntype is not None for ntype in self._ntypes
        ), "The node types have invalid IDs."
        for c_etype, etype_id in etypes.items():
            assert isinstance(c_etype, tuple) and len(c_etype) == 3, (
                "Expect canonical edge type in a triplet of string, but got "
                f"{c_etype}."
            )
            etype = c_etype[1]
            self._etypes[etype_id] = etype
            self._canonical_etypes[etype_id] = c_etype
            if etype in self._etype2canonical:
                # If one etype maps to multiple canonical etypes, empty tuple
                # is used to indicate such ambiguity casued by etype. See more
                # details in self.to_canonical_etype().
                self._etype2canonical[etype] = tuple()
            else:
                self._etype2canonical[etype] = c_etype
        assert all(
            etype is not None for etype in self._etypes
        ), "The edge types have invalid IDs."

        # This stores the node ID ranges for each node type in each partition.
        # The key is the node type, the value is a NumPy matrix with two
        # columns, in which each row indicates the start and the end of the
        # node ID range in a partition. The node IDs are global node IDs in the
        # homogeneous representation.
        self._typed_nid_range = {}
        # This stores the node ID map for per-node-type IDs in each partition.
        # The key is the node type, the value is a NumPy vector which indicates
        # the last node ID in a partition.
        self._typed_max_node_ids = {}
        max_node_map = np.zeros((num_parts,), dtype=np.int64)
        for key in node_map:
            assert key in ntypes, "Unexpected ntype: {}.".format(key)
            if not isinstance(node_map[key], np.ndarray):
                node_map[key] = F.asnumpy(node_map[key])
            assert node_map[key].shape == (num_parts, 2)
            self._typed_nid_range[key] = node_map[key]
            # This is used for per-node-type lookup.
            self._typed_max_node_ids[key] = np.cumsum(
                self._typed_nid_range[key][:, 1]
                - self._typed_nid_range[key][:, 0]
            )
            # This is used for homogeneous node ID lookup.
            max_node_map = np.maximum(
                self._typed_nid_range[key][:, 1], max_node_map
            )
        # This is a vector that indicates the last node ID in each partition.
        # The ID is the global ID in the homogeneous representation.
        self._max_node_ids = max_node_map

        # Similar to _typed_nid_range.
        self._typed_eid_range = {}
        # similar to _typed_max_node_ids.
        self._typed_max_edge_ids = {}
        max_edge_map = np.zeros((num_parts,), dtype=np.int64)
        for key in edge_map:
            assert key in etypes, "Unexpected etype: {}.".format(key)
            if not isinstance(edge_map[key], np.ndarray):
                edge_map[key] = F.asnumpy(edge_map[key])
            assert edge_map[key].shape == (num_parts, 2)
            self._typed_eid_range[key] = edge_map[key]
            # This is used for per-edge-type lookup.
            self._typed_max_edge_ids[key] = np.cumsum(
                self._typed_eid_range[key][:, 1]
                - self._typed_eid_range[key][:, 0]
            )
            # This is used for homogeneous edge ID lookup.
            max_edge_map = np.maximum(
                self._typed_eid_range[key][:, 1], max_edge_map
            )
        # Similar to _max_node_ids
        self._max_edge_ids = max_edge_map

        # These two are map functions that map node/edge IDs to node/edge type IDs.
        self._nid_map = IdMap(self._typed_nid_range)
        self._eid_map = IdMap(self._typed_eid_range)

        # Local node/edge type offset that maps the local homogenized node/edge IDs
        # to local heterogenized node/edge IDs.  One can do the mapping by binary search
        # on these arrays.
        self._local_ntype_offset = np.cumsum(
            [0]
            + [
                v[self._partid, 1] - v[self._partid, 0]
                for v in self._typed_nid_range.values()
            ]
        ).tolist()
        self._local_etype_offset = np.cumsum(
            [0]
            + [
                v[self._partid, 1] - v[self._partid, 0]
                for v in self._typed_eid_range.values()
            ]
        ).tolist()

        # Get meta data of the partition book
        self._partition_meta_data = []
        for partid in range(self._num_partitions):
            nrange_start = max_node_map[partid - 1] if partid > 0 else 0
            nrange_end = max_node_map[partid]
            num_nodes = nrange_end - nrange_start

            erange_start = max_edge_map[partid - 1] if partid > 0 else 0
            erange_end = max_edge_map[partid]
            num_edges = erange_end - erange_start

            part_info = {}
            part_info["machine_id"] = partid
            part_info["num_nodes"] = int(num_nodes)
            part_info["num_edges"] = int(num_edges)
            self._partition_meta_data.append(part_info)

    def shared_memory(self, graph_name):
        """Move data to shared memory."""
        # we need to store the nid ranges and eid ranges of different types in the order defined
        # by type IDs.
        nid_range = [None] * len(self.ntypes)
        for i, ntype in enumerate(self.ntypes):
            nid_range[i] = (ntype, self._typed_nid_range[ntype])
        nid_range_pickle = list(pickle.dumps(nid_range))

        eid_range = [None] * len(self.canonical_etypes)
        for i, etype in enumerate(self.canonical_etypes):
            eid_range[i] = (etype, self._typed_eid_range[etype])
        eid_range_pickle = list(pickle.dumps(eid_range))

        self._meta = _move_metadata_to_shared_mem(
            graph_name,
            0,  # We don't need to provide the number of nodes
            0,  # We don't need to provide the number of edges
            self._partid,
            self._num_partitions,
            F.tensor(nid_range_pickle),
            F.tensor(eid_range_pickle),
            True,
        )

    def num_partitions(self):
        """Return the number of partitions."""
        return self._num_partitions

    def _num_nodes(self, ntype=DEFAULT_NTYPE):
        """The total number of nodes"""
        if ntype == DEFAULT_NTYPE:
            return int(self._max_node_ids[-1])
        else:
            return int(self._typed_max_node_ids[ntype][-1])

    def _num_edges(self, etype=DEFAULT_ETYPE):
        """The total number of edges"""
        if etype in (DEFAULT_ETYPE, DEFAULT_ETYPE[1]):
            return int(self._max_edge_ids[-1])
        else:
            c_etype = self.to_canonical_etype(etype)
            return int(self._typed_max_edge_ids[c_etype][-1])

    def metadata(self):
        """Return the partition meta data."""
        return self._partition_meta_data

    def map_to_per_ntype(self, ids):
        """Map global homogeneous node IDs to node type IDs.
        Returns
            type_ids, per_type_ids
        """
        return self._nid_map(ids)

    def map_to_per_etype(self, ids):
        """Map global homogeneous edge IDs to edge type IDs.
        Returns
            type_ids, per_type_ids
        """
        return self._eid_map(ids)

    def map_to_homo_nid(self, ids, ntype):
        """Map per-node-type IDs to global node IDs in the homogeneous format."""
        ids = utils.toindex(ids).tousertensor()
        partids = self.nid2partid(ids, ntype)
        typed_max_nids = F.zerocopy_from_numpy(self._typed_max_node_ids[ntype])
        end_diff = F.gather_row(typed_max_nids, partids) - ids
        typed_nid_range = F.zerocopy_from_numpy(
            self._typed_nid_range[ntype][:, 1]
        )
        return F.gather_row(typed_nid_range, partids) - end_diff

    def map_to_homo_eid(self, ids, etype):
        """Map per-edge-type IDs to global edge IDs in the homoenegeous format."""
        ids = utils.toindex(ids).tousertensor()
        c_etype = self.to_canonical_etype(etype)
        partids = self.eid2partid(ids, c_etype)
        typed_max_eids = F.zerocopy_from_numpy(
            self._typed_max_edge_ids[c_etype]
        )
        end_diff = F.gather_row(typed_max_eids, partids) - ids
        typed_eid_range = F.zerocopy_from_numpy(
            self._typed_eid_range[c_etype][:, 1]
        )
        return F.gather_row(typed_eid_range, partids) - end_diff

    def nid2partid(self, nids, ntype=DEFAULT_NTYPE):
        """From global node IDs to partition IDs"""
        nids = utils.toindex(nids)
        if ntype == DEFAULT_NTYPE:
            ret = np.searchsorted(
                self._max_node_ids, nids.tonumpy(), side="right"
            )
        else:
            ret = np.searchsorted(
                self._typed_max_node_ids[ntype], nids.tonumpy(), side="right"
            )
        ret = utils.toindex(ret)
        return ret.tousertensor()

    def eid2partid(self, eids, etype=DEFAULT_ETYPE):
        """From global edge IDs to partition IDs"""
        eids = utils.toindex(eids)
        if etype in (DEFAULT_ETYPE, DEFAULT_ETYPE[1]):
            ret = np.searchsorted(
                self._max_edge_ids, eids.tonumpy(), side="right"
            )
        else:
            c_etype = self.to_canonical_etype(etype)
            ret = np.searchsorted(
                self._typed_max_edge_ids[c_etype], eids.tonumpy(), side="right"
            )
        ret = utils.toindex(ret)
        return ret.tousertensor()

    def partid2nids(self, partid, ntype=DEFAULT_NTYPE):
        """From partition ID to global node IDs"""
        # TODO do we need to cache it?
        if ntype == DEFAULT_NTYPE:
            start = self._max_node_ids[partid - 1] if partid > 0 else 0
            end = self._max_node_ids[partid]
            return F.arange(start, end)
        else:
            start = (
                self._typed_max_node_ids[ntype][partid - 1] if partid > 0 else 0
            )
            end = self._typed_max_node_ids[ntype][partid]
            return F.arange(start, end)

    def partid2eids(self, partid, etype=DEFAULT_ETYPE):
        """From partition ID to global edge IDs"""
        # TODO do we need to cache it?
        if etype in (DEFAULT_ETYPE, DEFAULT_ETYPE[1]):
            start = self._max_edge_ids[partid - 1] if partid > 0 else 0
            end = self._max_edge_ids[partid]
            return F.arange(start, end)
        else:
            c_etype = self.to_canonical_etype(etype)
            start = (
                self._typed_max_edge_ids[c_etype][partid - 1]
                if partid > 0
                else 0
            )
            end = self._typed_max_edge_ids[c_etype][partid]
            return F.arange(start, end)

    def nid2localnid(self, nids, partid, ntype=DEFAULT_NTYPE):
        """Get local node IDs within the given partition."""
        if partid != self._partid:
            raise RuntimeError(
                "Now RangePartitionBook does not support \
                getting remote tensor of nid2localnid."
            )

        nids = utils.toindex(nids)
        nids = nids.tousertensor()
        if ntype == DEFAULT_NTYPE:
            start = self._max_node_ids[partid - 1] if partid > 0 else 0
        else:
            start = (
                self._typed_max_node_ids[ntype][partid - 1] if partid > 0 else 0
            )
        return nids - int(start)

    def eid2localeid(self, eids, partid, etype=DEFAULT_ETYPE):
        """Get the local edge IDs within the given partition."""
        if partid != self._partid:
            raise RuntimeError(
                "Now RangePartitionBook does not support \
                getting remote tensor of eid2localeid."
            )

        eids = utils.toindex(eids)
        eids = eids.tousertensor()
        if etype in (DEFAULT_ETYPE, DEFAULT_ETYPE[1]):
            start = self._max_edge_ids[partid - 1] if partid > 0 else 0
        else:
            c_etype = self.to_canonical_etype(etype)
            start = (
                self._typed_max_edge_ids[c_etype][partid - 1]
                if partid > 0
                else 0
            )
        return eids - int(start)

    @property
    def partid(self):
        """Get the current partition ID."""
        return self._partid

    @property
    def ntypes(self):
        """Get the list of node types"""
        return self._ntypes

    @property
    def etypes(self):
        """Get the list of edge types"""
        return self._etypes

    @property
    def canonical_etypes(self):
        """Get the list of canonical edge types

        Returns
        -------
        list[(str, str, str)] or list[None]
            A list of canonical etypes. If keys of ``edge_map`` and ``etypes``
            are strings, a list of ``None`` is returned as canonical etypes
            are not available.
        """
        return self._canonical_etypes

    @property
    def local_ntype_offset(self):
        """Get the node type offset array of the local partition.

        The i-th element indicates the starting position of the i-th node type.
        """
        return self._local_ntype_offset

    @property
    def local_etype_offset(self):
        """Get the edge type offset array of the local partition.

        The i-th element indicates the starting position of the i-th edge type.
        """
        return self._local_etype_offset

    def to_canonical_etype(self, etype):
        """Convert an edge type to the corresponding canonical edge type.

        Parameters
        ----------
        etype : str or (str, str, str)
            The edge type

        Returns
        -------
        (str, str, str)
            The corresponding canonical edge type
        """
        if isinstance(etype, tuple):
            if etype not in self.canonical_etypes:
                raise DGLError('Edge type "{}" does not exist.'.format(etype))
            return etype
        ret = self._etype2canonical.get(etype, None)
        if ret is None:
            raise DGLError('Edge type "{}" does not exist.'.format(etype))
        if len(ret) == 0:
            raise DGLError(
                'Edge type "%s" is ambiguous. Please use canonical edge type '
                "in the form of (srctype, etype, dsttype)" % etype
            )
        return ret

    @property
    def global_nid_dtype(self):
        """Get the node ID's dtype"""
        return self._nid_map.torch_dtype

    @property
    def global_eid_dtype(self):
        """Get the edge ID's dtype"""
        return self._eid_map.torch_dtype


NODE_PART_POLICY = "node"
EDGE_PART_POLICY = "edge"
POLICY_DELIMITER = "~"


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
        Partition policy name, e.g., 'edge~_N:_E:_N' or 'node~_N'.
    partition_book : GraphPartitionBook
        A graph partition book
    """

    def __init__(self, policy_str, partition_book):
        assert policy_str.startswith(NODE_PART_POLICY) or policy_str.startswith(
            EDGE_PART_POLICY
        ), (
            f"policy_str must start with {NODE_PART_POLICY} or "
            f"{EDGE_PART_POLICY}, but got {policy_str}."
        )
        if NODE_PART_POLICY == policy_str:
            policy_str = NODE_PART_POLICY + POLICY_DELIMITER + DEFAULT_NTYPE
        if EDGE_PART_POLICY == policy_str:
            policy_str = EDGE_PART_POLICY + POLICY_DELIMITER + DEFAULT_ETYPE[1]
        self._policy_str = policy_str
        self._part_id = partition_book.partid
        self._partition_book = partition_book
        part_policy, self._type_name = policy_str.split(POLICY_DELIMITER, 1)
        if part_policy == EDGE_PART_POLICY:
            self._type_name = _etype_str_to_tuple(self._type_name)
        self._is_node = self.policy_str.startswith(NODE_PART_POLICY)

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
    def type_name(self):
        """Get the type name: ntype or etype

        Returns
        -------
        str or (str, str, str)
            The ntype or etype.
        """
        return self._type_name

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

    @property
    def is_node(self):
        """Indicate whether the policy is for node or edge

        Returns
        -------
        bool
            node or edge
        """
        return self._is_node

    def get_data_name(self, name):
        """Get HeteroDataName"""
        return HeteroDataName(self.is_node, self.type_name, name)

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
        if self.is_node:
            return self._partition_book.nid2localnid(
                id_tensor, self._part_id, self.type_name
            )
        else:
            return self._partition_book.eid2localeid(
                id_tensor, self._part_id, self.type_name
            )

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
        if self.is_node:
            return self._partition_book.nid2partid(id_tensor, self.type_name)
        else:
            return self._partition_book.eid2partid(id_tensor, self.type_name)

    def get_part_size(self):
        """Get data size of current partition.

        Returns
        -------
        int
            data size
        """
        if self.is_node:
            return len(
                self._partition_book.partid2nids(self._part_id, self.type_name)
            )
        else:
            return len(
                self._partition_book.partid2eids(self._part_id, self.type_name)
            )

    def get_size(self):
        """Get the full size of the data.

        Returns
        -------
        int
            data size
        """
        if self.is_node:
            return self._partition_book._num_nodes(self.type_name)
        else:
            return self._partition_book._num_edges(self.type_name)


class NodePartitionPolicy(PartitionPolicy):
    """Partition policy for nodes."""

    def __init__(self, partition_book, ntype=DEFAULT_NTYPE):
        super(NodePartitionPolicy, self).__init__(
            NODE_PART_POLICY + POLICY_DELIMITER + ntype, partition_book
        )


class EdgePartitionPolicy(PartitionPolicy):
    """Partition policy for edges."""

    def __init__(self, partition_book, etype=DEFAULT_ETYPE):
        assert (
            isinstance(etype, tuple) and len(etype) == 3
        ), f"Expect canonical edge type in a triplet of string, but got {etype}."
        super(EdgePartitionPolicy, self).__init__(
            EDGE_PART_POLICY + POLICY_DELIMITER + _etype_tuple_to_str(etype),
            partition_book,
        )


class HeteroDataName(object):
    """The data name in a heterogeneous graph.

    A unique data name has three components:
    * indicate it's node data or edge data.
    * indicate the node/edge type.
    * the name of the data.

    Parameters
    ----------
    is_node : bool
        Indicate whether it's node data or edge data.
    entity_type : str or (str, str, str)
        The type of the node/edge.
    data_name : str
        The name of the data.
    """

    def __init__(self, is_node, entity_type, data_name):
        self._policy = NODE_PART_POLICY if is_node else EDGE_PART_POLICY
        if not is_node:
            assert isinstance(entity_type, tuple) and len(entity_type) == 3, (
                "Expect canonical edge type in a triplet of string, but got "
                f"{entity_type}."
            )
        self._entity_type = entity_type
        self.data_name = data_name

    @property
    def policy_str(self):
        """concatenate policy and entity type into string"""
        entity_type = self.get_type()
        if self.is_edge():
            entity_type = _etype_tuple_to_str(entity_type)
        return self._policy + POLICY_DELIMITER + entity_type

    def is_node(self):
        """Is this the name of node data"""
        return self._policy == NODE_PART_POLICY

    def is_edge(self):
        """Is this the name of edge data"""
        return self._policy == EDGE_PART_POLICY

    def get_type(self):
        """The type of the node/edge.
        This is only meaningful in a heterogeneous graph.
        In homogeneous graph, type is '_N' for a node and '_N:_E:_N' for an
        edge.
        """
        return self._entity_type

    def get_name(self):
        """The name of the data."""
        return self.data_name

    def __str__(self):
        """The full name of the data.

        The full name is used as the key in the KVStore.
        """
        return self.policy_str + POLICY_DELIMITER + self.data_name


def parse_hetero_data_name(name):
    """Parse data name and create HeteroDataName.

    The data name has a specialized format. We can parse the name to determine if
    it's node data or edge data, node/edge type and its actual name. The data name
    has three fields and they are separated by ":".

    Parameters
    ----------
    name : str
        The data name

    Returns
    -------
    HeteroDataName
    """
    names = name.split(POLICY_DELIMITER)
    assert len(names) == 3, "{} is not a valid heterograph data name".format(
        name
    )
    assert names[0] in (
        NODE_PART_POLICY,
        EDGE_PART_POLICY,
    ), "{} is not a valid heterograph data name".format(name)
    is_node = names[0] == NODE_PART_POLICY
    entity_type = names[1]
    if not is_node:
        entity_type = _etype_str_to_tuple(entity_type)
    return HeteroDataName(is_node, entity_type, names[2])
