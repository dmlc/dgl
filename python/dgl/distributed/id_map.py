"""Module for mapping between node/edge IDs and node/edge types."""
import numpy as np

from .._ffi.function import _init_api
from .. import backend as F
from .. import utils

class IdMap:
    '''A map for converting node/edge IDs to their type IDs and type-wise IDs.

    For a heterogeneous graph, DGL assigns an integer ID to each node/edge type;
    node and edge of different types have independent IDs starting from zero.
    Therefore, a node/edge can be uniquely identified by an ID pair,
    ``(type_id, type_wise_id)``. To make it convenient for distributed processing,
    DGL further encodes the ID pair into one integer ID, which we refer to
    as *homogeneous ID*.

    DGL arranges nodes and edges so that all nodes of the same type have contiguous
    homogeneous IDs. If the graph is partitioned, the nodes/edges of the same type
    within a partition have contiguous homogeneous IDs.

    Below is an example adjancency matrix of an unpartitioned heterogeneous graph
    stored using the above ID assignment. Here, the graph has two types of nodes
    (``T0`` and ``T1``), and four types of edges (``R0``, ``R1``, ``R2``, ``R3``).
    There are a total of 400 nodes in the graph and each type has 200 nodes. Nodes
    of type 0 have IDs in [0,200), while nodes of type 1 have IDs in [200, 400).

    ```
        0 <- T0 -> 200 <- T1 -> 400
     0  +-----------+------------+
        |           |            |
     ^  |    R0     |     R1     |
     T0 |           |            |
     v  |           |            |
    200 +-----------+------------+
        |           |            |
     ^  |    R2     |     R3     |
     T1 |           |            |
     v  |           |            |
    400 +-----------+------------+
    ```

    Below shows the adjacency matrix after the graph is partitioned into two.
    Note that each partition still has two node types and four edge types,
    and nodes/edges of the same type have contiguous IDs.

    ```
                partition 0              partition 1

        0 <- T0 -> 100 <- T1 -> 200 <- T0 -> 300 <- T1 -> 400
     0  +-----------+------------+-----------+------------+
        |           |            |                        |
     ^  |    R0     |     R1     |                        |
     T0 |           |            |                        |
     v  |           |            |                        |
    100 +-----------+------------+                        |
        |           |            |                        |
     ^  |    R2     |     R3     |                        |
     T1 |           |            |                        |
     v  |           |            |                        |
    200 +-----------+------------+-----------+------------+
        |                        |           |            |
     ^  |                        |    R0     |     R1     |
     T0 |                        |           |            |
     v  |                        |           |            |
    100 |                        +-----------+------------+
        |                        |           |            |
     ^  |                        |    R2     |     R3     |
     T1 |                        |           |            |
     v  |                        |           |            |
    200 +-----------+------------+-----------+------------+
    ```

    The following table is an alternative way to represent the above ID assignments.
    It is easy to see that the homogeneous ID range [0, 100) is used for nodes of type 0
    in partition 0, [100, 200) is used for nodes of type 1 in partition 0, and so on.
    ```
    +---------+------+----------
      range   | type | partition
    [0, 100)  |   0  |    0
    [100,200) |   1  |    0
    [200,300) |   0  |    1
    [300,400) |   1  |    1
    ```

    The goal of this class is to, given a node's homogenous ID, convert it into the
    ID pair ``(type_id, type_wise_id)``. For example, homogeneous node ID 90 is mapped
    to (0, 90); homogeneous node ID 201 is mapped to (0, 101).

    Parameters
    ----------
    id_ranges : dict[str, Tensor].
        Node ID ranges within partitions for each node type. The key is the node type
        name in string. The value is a tensor of shape :math:`(K, 2)`, where :math:`K` is
        the number of partitions. Each row has two integers: the starting and the ending IDs
        for a particular node type in a partition. For example, all nodes of type ``"T"`` in
        partition ``i`` has ID range ``id_ranges["T"][i][0]`` to ``id_ranges["T"][i][1]``.
        It is the same as the `node_map` argument in `RangePartitionBook`.
    '''
    def __init__(self, id_ranges):
        self.num_parts = list(id_ranges.values())[0].shape[0]
        self.num_types = len(id_ranges)
        ranges = np.zeros((self.num_parts * self.num_types, 2), dtype=np.int64)
        typed_map = []
        id_ranges = list(id_ranges.values())
        id_ranges.sort(key=lambda a: a[0, 0])
        for i, id_range in enumerate(id_ranges):
            ranges[i::self.num_types] = id_range
            map1 = np.cumsum(id_range[:, 1] - id_range[:, 0])
            typed_map.append(map1)

        assert np.all(np.diff(ranges[:, 0]) >= 0)
        assert np.all(np.diff(ranges[:, 1]) >= 0)
        self.range_start = utils.toindex(np.ascontiguousarray(ranges[:, 0]))
        self.range_end = utils.toindex(np.ascontiguousarray(ranges[:, 1]) - 1)
        self.typed_map = utils.toindex(np.concatenate(typed_map))

    def __call__(self, ids):
        '''Convert the homogeneous IDs to (type_id, type_wise_id).

        Parameters
        ----------
        ids : 1D tensor
            The homogeneous ID.

        Returns
        -------
        type_ids : Tensor
            Type IDs
        per_type_ids : Tensor
            Type-wise IDs
        '''
        if self.num_types == 0:
            return F.zeros((len(ids),), F.dtype(ids), F.cpu()), ids
        if len(ids) == 0:
            return ids, ids

        ids = utils.toindex(ids)
        ret = _CAPI_DGLHeteroMapIds(ids.todgltensor(),
                                    self.range_start.todgltensor(),
                                    self.range_end.todgltensor(),
                                    self.typed_map.todgltensor(),
                                    self.num_parts, self.num_types)
        ret = utils.toindex(ret).tousertensor()
        return ret[:len(ids)], ret[len(ids):]

_init_api("dgl.distributed.id_map")
