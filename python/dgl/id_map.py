"""Module for mapping between node/edge Ids and node/edge types."""
import numpy as np

from ._ffi.function import _init_api
from . import backend as F
from . import utils

class IdMap:
    '''This maps node/edge Ids in the homogeneous form to per-type form.

    It stores the Id ranges for each partition and for each type. The Ids are in
    the homogeneous form (i.e., all nodes of different types have unique Ids). For each type,
    all Ids in a partition fall in a contiguous range. The Id ranges are stored in
    a matrix, where each row has two elements to represent the beginning and the end of
    the range.

    This class computes the homogeneous Ids to per-type Ids and their types.

    Parameters
    ----------
    id_ranges : dict of tensors.
        The Id ranges of each partition for each type.
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
        '''Map Ids in the homogeneous form to per-type Ids and types.

        Parameters
        ----------
        ids : 1D tensor
            The homogeneous Id.

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
