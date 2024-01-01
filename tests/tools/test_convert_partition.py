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


def test_get_unique_invidx_low_mem():
    srcids = np.array([14, 0, 3, 3, 0, 3, 9, 5, 14, 12])
    dstids = np.array([10, 16, 12, 13, 10, 17, 16, 13, 14, 16])
    unique_nids = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    uniques, idxes, srcids, dstids = _get_unique_invidx(
        srcids,
        dstids,
        unique_nids,
        low_mem=True,
    )
    expected_unqiues = np.array(
        [0, 3, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    )
    expected_idxes = np.array(
        [1, 2, 7, 6, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    )
    expected_srcids = np.array([8, 0, 1, 1, 0, 1, 3, 2, 8, 6])
    expected_dstids = np.array([4, 10, 6, 7, 4, 11, 10, 7, 8, 10])
    assert np.all(
        uniques == expected_unqiues
    ), f"unique is not expected. {uniques} != {expected_unqiues}"
    assert np.all(
        idxes == expected_idxes
    ), f"indices is not expected. {idxes} != {expected_idxes}"
    assert np.all(
        srcids == expected_srcids
    ), f"srcids is not expected. {srcids} != {expected_srcids}"
    assert np.all(
        dstids == expected_dstids
    ), f"dstdis is not expected. {dstids} != {expected_dstids}"


def test_get_unique_invidx_high_mem():
    srcids = np.array([14, 0, 3, 3, 0, 3, 9, 5, 14, 12])
    dstids = np.array([10, 16, 12, 13, 10, 17, 16, 13, 14, 16])
    unique_nids = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    uniques, idxes, srcids, dstids = _get_unique_invidx(
        srcids,
        dstids,
        unique_nids,
        low_mem=False,
    )
    expected_unqiues = np.array(
        [0, 3, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    )
    expected_idxes = np.array(
        [1, 2, 7, 6, 10, 21, 9, 13, 0, 25, 11, 15, 28, 29]
    )
    expected_srcids = np.array([8, 0, 1, 1, 0, 1, 3, 2, 8, 6])
    expected_dstids = np.array([4, 10, 6, 7, 4, 11, 10, 7, 8, 10])
    assert np.all(
        uniques == expected_unqiues
    ), f"unique is not expected. {uniques} != {expected_unqiues}"
    assert np.all(
        idxes == expected_idxes
    ), f"indices is not expected. {idxes} != {expected_idxes}"
    assert np.all(
        srcids == expected_srcids
    ), f"srcids is not expected. {srcids} != {expected_srcids}"
    assert np.all(
        dstids == expected_dstids
    ), f"dstdis is not expected. {dstids} != {expected_dstids}"


def test_get_unique_invidx_low_high_mem():
    srcids = np.array([14, 0, 3, 3, 0, 3, 9, 5, 14, 12])
    dstids = np.array([10, 16, 12, 13, 10, 17, 16, 13, 14, 16])
    unique_nids = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    uniques_low, idxes_low, srcids_low, dstids_low = _get_unique_invidx(
        srcids,
        dstids,
        unique_nids,
        low_mem=True,
    )
    uniques_high, idxes_high, srcids_high, dstids_high = _get_unique_invidx(
        srcids,
        dstids,
        unique_nids,
        low_mem=False,
    )
    assert np.all(
        uniques_low == uniques_high
    ), f"unique is not expected. {uniques_low} != {uniques_high}"
    assert not np.all(
        idxes_low == idxes_high
    ), f"indices is not expected. {idxes_low} == {idxes_high}"
    assert np.all(
        srcids_low == srcids_high
    ), f"srcids is not expected. {srcids_low} != {srcids_high}"
    assert np.all(
        dstids_low == dstids_high
    ), f"dstdis is not expected. {dstids_low} != {dstids_high}"
