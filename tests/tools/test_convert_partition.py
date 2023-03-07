import os
import tempfile

import numpy as np
import pytest

from tools.distpartitioning import utils

from tools.distpartitioning.convert_partition import _get_unique_invidx


@pytest.mark.parametrize(
    "num_nodes, num_edges, nid_begin, nid_end",
    [
        [4000, 40000, 0, 1000],
        [4000, 40000, 1000, 2000],
        [4000, 40000, 2000, 3000],
        [4000, 40000, 3000, 4000],
        [1, 1, 0, 1],
    ],
)
def test_get_unique_invidx(num_nodes, num_edges, nid_begin, nid_end):
    # prepare data for the function
    # generate synthetic edges
    if num_edges > 0:
        srcids = np.random.randint(0, num_nodes, (num_edges,))
        dstids = np.random.randint(nid_begin, nid_end, (num_edges,))
    else:
        srcids = np.array([])
        dstids = np.array([])

    assert nid_begin <= nid_end

    # generate unique node-ids for any
    # partition. This list should be sorted.
    # This is equivilant to shuffle_nids in a partition
    unique_nids = np.arange(nid_begin, nid_end)

    # invoke the test target
    uniques, idxes, src_ids, dst_ids = _get_unique_invidx(
        srcids, dstids, unique_nids
    )

    # validate the outputs of this function
    # array uniques should be sorted list of integers.
    assert np.all(
        np.diff(uniques) >= 0
    ), f"Output parameter uniques assert failing."

    # idxes are list of integers
    # these are indices in the concatenated list (srcids, dstids, unique_nids)
    max_idx = len(src_ids) + len(dst_ids) + len(unique_nids)
    assert np.all(idxes >= 0), f"Output parameter idxes has negative values."
    assert np.all(
        idxes < max_idx
    ), f"Output parameter idxes has invalid maximum value."

    # srcids and dstids will be inverse indices in the uniques list
    min_src = np.amin(src_ids)
    max_src = np.amax(src_ids)

    min_dst = np.amin(dst_ids)
    max_dst = np.amax(dst_ids)

    assert (
        len(uniques) > max_src
    ), f"Inverse idx, src_ids, has invalid max value."
    assert min_src >= 0, f"Inverse idx, src_ids has negative values."

    assert len(uniques) > max_dst, f"Inverse idx, dst_ids, invalid max value."
    assert max_dst >= 0, f"Inverse idx, dst_ids has negative values."
