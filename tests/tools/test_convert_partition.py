import os
import tempfile

import numpy as np
import pytest

import utils

from convert_partition import _get_unique_invidx


@pytest.mark.parametrize(
    "num_nodes, num_edges, nid_begin, nid_end",
    [
        [4000, 40000, 0, 1000],
        [4000, 40000, 1000, 2000],
        [4000, 40000, 2000, 3000],
        [4000, 40000, 3000, 4000],
        [4000, 100, 0, 1000],
        [4000, 100, 1000, 2000],
        [4000, 100, 2000, 3000],
        [4000, 100, 3000, 4000],
        [1, 1, 0, 1],
    ],
)
def test_get_unique_invidx_with_numpy(num_nodes, num_edges, nid_begin, nid_end):
    # prepare data for the function
    # generate synthetic edges
    if num_edges > 0:
        srcids = np.random.randint(0, num_nodes, (num_edges,))  # exclusive
        dstids = np.random.randint(
            nid_begin, nid_end, (num_edges,)
        )  # exclusive
    else:
        srcids = np.array([])
        dstids = np.array([])

    assert nid_begin <= nid_end

    # generate unique node-ids for any
    # partition. This list should be sorted.
    # This is equivilant to shuffle_nids in a partition
    unique_nids = np.arange(nid_begin, nid_end)  # exclusive

    # test with numpy unique here
    orig_srcids = srcids.copy()
    orig_dstids = dstids.copy()
    input_arr = np.concatenate([srcids, dstids, unique_nids])

    # test
    uniques, idxes, srcids, dstids = _get_unique_invidx(
        srcids, dstids, unique_nids
    )

    assert len(uniques) == len(idxes)
    assert np.all(srcids < len(uniques))
    assert np.all(dstids < len(uniques))
    assert np.all(uniques[srcids].sort() == orig_srcids.sort())
    assert np.all(uniques[dstids] == orig_dstids)

    assert np.all(uniques == input_arr[idxes])

    # numpy
    np_uniques, np_idxes, np_inv_idxes = np.unique(
        np.concatenate([orig_srcids, orig_dstids, unique_nids]),
        return_index=True,
        return_inverse=True,
    )

    # test uniques
    assert np.all(np_uniques == uniques)

    # test idxes array
    assert np.all(input_arr[idxes].sort() == input_arr[np_idxes].sort())

    # test srcids, inv_indices
    assert np.all(
        uniques[srcids].sort()
        == np_uniques[np_inv_idxes[0 : len(srcids)]].sort()
    )

    # test dstids, inv_indices
    assert np.all(
        uniques[dstids].sort() == np_uniques[np_inv_idxes[len(srcids) :]].sort()
    )


@pytest.mark.parametrize(
    "num_nodes, num_edges, nid_begin, nid_end",
    [
        # dense networks, no. of edges more than no. of nodes
        [4000, 40000, 0, 1000],
        [4000, 40000, 1000, 2000],
        [4000, 40000, 2000, 3000],
        [4000, 40000, 3000, 4000],
        # sparse networks, no. of edges smaller than no. of nodes
        [4000, 100, 0, 1000],
        [4000, 100, 1000, 2000],
        [4000, 100, 2000, 3000],
        [4000, 100, 3000, 4000],
        # corner case
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
