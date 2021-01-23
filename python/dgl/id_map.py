"""Module for mapping between node/edge IDs and node/edge types."""
import numpy as np

from ._ffi.function import _init_api
from . import backend as F
from . import utils

class IdMap:
    '''Map node/edge IDs in the homogeneous form to per-type IDs and determine their types.

    For a heterogeneous graph, we can use a homogeneous graph format to store it. With
    this format, all nodes have unique IDs. We refer to these IDs as homogeneous IDs.
    In a heterogeneous graph format, IDs are only unique within the type. We refer to
    this type of IDs as per-type IDs.

    When storing a heterogeneous graph in the homogeneous format, we arrange node IDs
    and edge IDs so that all nodes of the same type have contiguous IDs. If
    the graph is partitioned, the nodes of the same type within a partition have
    contiguous IDs.

    The table below shows an example of storing node IDs in a heterogeneous graph
    in the homogeneous format. The heterogeneous graph has two node types and has
    two partitions. The homogeneous ID range [0, 100) is used for nodes of type 0
    in partition 0, [100, 200) is used for nodes of type 1 in partition 0, and so on.

    ```
    +---------+------+----------
      range   | type | partition
    [0, 100)  |   0  |    0
    [100,200) |   1  |    0
    [200,300) |   0  |    1
    [300,400) |   1  |    1
    ```

    This class is to map any homogeneous ID to a per-type ID and its type. For example,
    homogeneous node ID 90 is mapped to type 0 and per-type ID 90; homogeneous node ID
    201 is mapped to type 0 and per-type ID 101.

    Parameters
    ----------
    id_ranges : dict[str, Tensor].
        The ID ranges of each partition for each type. It stores the same information
        as the `node_map` argument in `RangePartitionBook`.
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
        '''Map IDs in the homogeneous form to per-type IDs and types.

        Parameters
        ----------
        ids : 1D tensor
            The homogeneous ID.

        Returns
        -------
            type_ids, per_type_ids
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

_init_api("dgl.id_map")
