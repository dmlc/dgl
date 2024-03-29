import os
import re
from sys import platform

import dgl
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dgl import graphbolt as gb
from torch.testing import assert_close


def test_ItemSampler_minibatcher():
    # Default minibatcher is used if not specified.
    # Warning message is raised if names are not specified.
    item_set = gb.ItemSet(torch.arange(0, 10))
    item_sampler = gb.ItemSampler(item_set, batch_size=4)
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Failed to map item list to `MiniBatch` as the names of items are "
            "not provided. Please provide a customized `MiniBatcher`. The "
            "item list is returned as is."
        ),
    ):
        minibatch = next(iter(item_sampler))
        assert not isinstance(minibatch, gb.MiniBatch)

    # Default minibatcher is used if not specified.
    # Warning message is raised if unrecognized names are specified.
    item_set = gb.ItemSet(torch.arange(0, 10), names="unknown_name")
    item_sampler = gb.ItemSampler(item_set, batch_size=4)
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Unknown item name 'unknown_name' is detected and added into "
            "`MiniBatch`. You probably need to provide a customized "
            "`MiniBatcher`."
        ),
    ):
        minibatch = next(iter(item_sampler))
        assert isinstance(minibatch, gb.MiniBatch)
        assert minibatch.unknown_name is not None

    # Default minibatcher is used if not specified.
    # `MiniBatch` is returned if expected names are specified.
    item_set = gb.ItemSet(torch.arange(0, 10), names="seed_nodes")
    item_sampler = gb.ItemSampler(item_set, batch_size=4)
    minibatch = next(iter(item_sampler))
    assert isinstance(minibatch, gb.MiniBatch)
    assert minibatch.seeds is not None
    assert len(minibatch.seeds) == 4

    # Customized minibatcher is used if specified.
    def minibatcher(batch, names):
        return gb.MiniBatch(seeds=batch)

    item_sampler = gb.ItemSampler(
        item_set, batch_size=4, minibatcher=minibatcher
    )
    minibatch = next(iter(item_sampler))
    assert isinstance(minibatch, gb.MiniBatch)
    assert minibatch.seeds is not None
    assert len(minibatch.seeds) == 4


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSet_Iterable_Only(batch_size, shuffle, drop_last):
    num_ids = 103

    class InvalidLength:
        def __iter__(self):
            return iter(torch.arange(0, num_ids))

    seed_nodes = gb.ItemSet(InvalidLength())
    item_set = gb.ItemSet(seed_nodes, names="seed_nodes")
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    minibatch_ids = []
    for i, minibatch in enumerate(item_sampler):
        assert isinstance(minibatch, gb.MiniBatch)
        assert minibatch.seeds is not None
        assert minibatch.labels is None
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            assert len(minibatch.seeds) == batch_size
        else:
            if not drop_last:
                assert len(minibatch.seeds) == num_ids % batch_size
            else:
                assert False
        minibatch_ids.append(minibatch.seeds)
    minibatch_ids = torch.cat(minibatch_ids)
    assert torch.all(minibatch_ids[:-1] <= minibatch_ids[1:]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSet_integer(batch_size, shuffle, drop_last):
    # Node IDs.
    num_ids = 103
    item_set = gb.ItemSet(num_ids, names="seed_nodes")
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    minibatch_ids = []
    for i, minibatch in enumerate(item_sampler):
        assert isinstance(minibatch, gb.MiniBatch)
        assert minibatch.seeds is not None
        assert minibatch.labels is None
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            assert len(minibatch.seeds) == batch_size
        else:
            if not drop_last:
                assert len(minibatch.seeds) == num_ids % batch_size
            else:
                assert False
        minibatch_ids.append(minibatch.seeds)
    minibatch_ids = torch.cat(minibatch_ids)
    assert torch.all(minibatch_ids[:-1] <= minibatch_ids[1:]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSet_seed_nodes(batch_size, shuffle, drop_last):
    # Node IDs.
    num_ids = 103
    seed_nodes = torch.arange(0, num_ids)
    item_set = gb.ItemSet(seed_nodes, names="seed_nodes")
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    minibatch_ids = []
    for i, minibatch in enumerate(item_sampler):
        assert isinstance(minibatch, gb.MiniBatch)
        assert minibatch.seeds is not None
        assert minibatch.labels is None
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            assert len(minibatch.seeds) == batch_size
        else:
            if not drop_last:
                assert len(minibatch.seeds) == num_ids % batch_size
            else:
                assert False
        minibatch_ids.append(minibatch.seeds)
    minibatch_ids = torch.cat(minibatch_ids)
    assert torch.all(minibatch_ids[:-1] <= minibatch_ids[1:]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSet_seed_nodes_labels(batch_size, shuffle, drop_last):
    # Node IDs.
    num_ids = 103
    seed_nodes = torch.arange(0, num_ids)
    labels = torch.arange(0, num_ids)
    item_set = gb.ItemSet((seed_nodes, labels), names=("seed_nodes", "labels"))
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    minibatch_ids = []
    minibatch_labels = []
    for i, minibatch in enumerate(item_sampler):
        assert isinstance(minibatch, gb.MiniBatch)
        assert minibatch.seeds is not None
        assert minibatch.labels is not None
        assert len(minibatch.seeds) == len(minibatch.labels)
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            assert len(minibatch.seeds) == batch_size
        else:
            if not drop_last:
                assert len(minibatch.seeds) == num_ids % batch_size
            else:
                assert False
        minibatch_ids.append(minibatch.seeds)
        minibatch_labels.append(minibatch.labels)
    minibatch_ids = torch.cat(minibatch_ids)
    minibatch_labels = torch.cat(minibatch_labels)
    assert torch.all(minibatch_ids[:-1] <= minibatch_ids[1:]) is not shuffle
    assert (
        torch.all(minibatch_labels[:-1] <= minibatch_labels[1:]) is not shuffle
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSet_graphs(batch_size, shuffle, drop_last):
    # Graphs.
    num_graphs = 103
    num_nodes = 10
    num_edges = 20
    graphs = [
        dgl.rand_graph(num_nodes * (i + 1), num_edges * (i + 1))
        for i in range(num_graphs)
    ]
    item_set = gb.ItemSet(graphs, names="graphs")
    # DGLGraph is not supported in gb.MiniBatch yet. Let's use a customized
    # minibatcher to return the original graphs.
    customized_minibatcher = lambda batch, names: batch
    item_sampler = gb.ItemSampler(
        item_set,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        minibatcher=customized_minibatcher,
    )
    minibatch_num_nodes = []
    minibatch_num_edges = []
    for i, minibatch in enumerate(item_sampler):
        is_last = (i + 1) * batch_size >= num_graphs
        if not is_last or num_graphs % batch_size == 0:
            assert minibatch.batch_size == batch_size
        else:
            if not drop_last:
                assert minibatch.batch_size == num_graphs % batch_size
            else:
                assert False
        minibatch_num_nodes.append(minibatch.batch_num_nodes())
        minibatch_num_edges.append(minibatch.batch_num_edges())
    minibatch_num_nodes = torch.cat(minibatch_num_nodes)
    minibatch_num_edges = torch.cat(minibatch_num_edges)
    assert (
        torch.all(minibatch_num_nodes[:-1] <= minibatch_num_nodes[1:])
        is not shuffle
    )
    assert (
        torch.all(minibatch_num_edges[:-1] <= minibatch_num_edges[1:])
        is not shuffle
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSet_node_pairs(batch_size, shuffle, drop_last):
    # Node pairs.
    num_ids = 103
    node_pairs = torch.arange(0, 2 * num_ids).reshape(-1, 2)
    item_set = gb.ItemSet(node_pairs, names="node_pairs")
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    src_ids = []
    dst_ids = []
    for i, minibatch in enumerate(item_sampler):
        assert minibatch.seeds is not None
        assert isinstance(minibatch.seeds, torch.Tensor)
        assert minibatch.labels is None
        src, dst = minibatch.seeds.T
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = num_ids % batch_size
            else:
                assert False
        assert len(src) == expected_batch_size
        assert len(dst) == expected_batch_size
        # Verify src and dst IDs match.
        assert torch.equal(src + 1, dst)
        # Archive batch.
        src_ids.append(src)
        dst_ids.append(dst)
    src_ids = torch.cat(src_ids)
    dst_ids = torch.cat(dst_ids)
    assert torch.all(src_ids[:-1] <= src_ids[1:]) is not shuffle
    assert torch.all(dst_ids[:-1] <= dst_ids[1:]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSet_node_pairs_labels(batch_size, shuffle, drop_last):
    # Node pairs and labels
    num_ids = 103
    node_pairs = torch.arange(0, 2 * num_ids).reshape(-1, 2)
    labels = node_pairs[:, 0]
    item_set = gb.ItemSet((node_pairs, labels), names=("node_pairs", "labels"))
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    src_ids = []
    dst_ids = []
    labels = []
    for i, minibatch in enumerate(item_sampler):
        assert minibatch.seeds is not None
        assert isinstance(minibatch.seeds, torch.Tensor)
        assert minibatch.labels is not None
        src, dst = minibatch.seeds.T
        label = minibatch.labels
        assert len(src) == len(dst)
        assert len(src) == len(label)
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = num_ids % batch_size
            else:
                assert False
        assert len(src) == expected_batch_size
        assert len(dst) == expected_batch_size
        assert len(label) == expected_batch_size
        # Verify src/dst IDs and labels match.
        assert torch.equal(src + 1, dst)
        assert torch.equal(src, label)
        # Archive batch.
        src_ids.append(src)
        dst_ids.append(dst)
        labels.append(label)
    src_ids = torch.cat(src_ids)
    dst_ids = torch.cat(dst_ids)
    labels = torch.cat(labels)
    assert torch.all(src_ids[:-1] <= src_ids[1:]) is not shuffle
    assert torch.all(dst_ids[:-1] <= dst_ids[1:]) is not shuffle
    assert torch.all(labels[:-1] <= labels[1:]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSet_node_pairs_negative_dsts(batch_size, shuffle, drop_last):
    # Node pairs and negative destinations.
    num_ids = 103
    num_negs = 2
    node_pairs = torch.arange(0, 2 * num_ids).reshape(-1, 2)
    neg_dsts = torch.arange(
        2 * num_ids, 2 * num_ids + num_ids * num_negs
    ).reshape(-1, num_negs)
    item_set = gb.ItemSet(
        (node_pairs, neg_dsts), names=("node_pairs", "negative_dsts")
    )
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    src_ids = []
    dst_ids = []
    negs_ids = []
    for i, minibatch in enumerate(item_sampler):
        assert minibatch.seeds is not None
        assert isinstance(minibatch.seeds, torch.Tensor)
        assert minibatch.labels is not None
        assert minibatch.indexes is not None
        src, dst = minibatch.seeds.T
        negs_src = src[~minibatch.labels.to(bool)]
        negs_dst = dst[~minibatch.labels.to(bool)]
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = num_ids % batch_size
            else:
                assert False
        assert len(src) == expected_batch_size * 3
        assert len(dst) == expected_batch_size * 3
        assert negs_src.dim() == 1
        assert negs_dst.dim() == 1
        assert len(negs_src) == expected_batch_size * 2
        assert len(negs_dst) == expected_batch_size * 2
        expected_indexes = torch.arange(expected_batch_size)
        expected_indexes = torch.cat(
            (expected_indexes, expected_indexes.repeat_interleave(2))
        )
        assert torch.equal(minibatch.indexes, expected_indexes)
        # Verify node pairs and negative destinations.
        assert torch.equal(
            src[minibatch.labels.to(bool)] + 1, dst[minibatch.labels.to(bool)]
        )
        assert torch.equal((negs_dst - 2 * num_ids) // 2 * 2, negs_src)
        # Archive batch.
        src_ids.append(src)
        dst_ids.append(dst)
        negs_ids.append(negs_dst)
    src_ids = torch.cat(src_ids)
    dst_ids = torch.cat(dst_ids)
    negs_ids = torch.cat(negs_ids)
    assert torch.all(src_ids[:-1] <= src_ids[1:]) is not shuffle
    assert torch.all(dst_ids[:-1] <= dst_ids[1:]) is not shuffle
    assert torch.all(negs_ids[:-1] <= negs_ids[1:]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSet_seeds(batch_size, shuffle, drop_last):
    # Node pairs.
    num_ids = 103
    seeds = torch.arange(0, 3 * num_ids).reshape(-1, 3)
    item_set = gb.ItemSet(seeds, names="seeds")
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    seeds_ids = []
    for i, minibatch in enumerate(item_sampler):
        assert minibatch.seeds is not None
        assert isinstance(minibatch.seeds, torch.Tensor)
        assert minibatch.labels is None
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = num_ids % batch_size
            else:
                assert False
        assert minibatch.seeds.shape == (expected_batch_size, 3)
        # Verify seeds match.
        assert torch.equal(minibatch.seeds[:, 0] + 1, minibatch.seeds[:, 1])
        assert torch.equal(minibatch.seeds[:, 1] + 1, minibatch.seeds[:, 2])
        # Archive batch.
        seeds_ids.append(minibatch.seeds)
    seeds_ids = torch.cat(seeds_ids)
    assert torch.all(seeds_ids[:-1, 0] <= seeds_ids[1:, 0]) is not shuffle
    assert torch.all(seeds_ids[:-1, 1] <= seeds_ids[1:, 1]) is not shuffle
    assert torch.all(seeds_ids[:-1, 2] <= seeds_ids[1:, 2]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSet_seeds_labels(batch_size, shuffle, drop_last):
    # Node pairs and labels
    num_ids = 103
    seeds = torch.arange(0, 3 * num_ids).reshape(-1, 3)
    labels = seeds[:, 0]
    item_set = gb.ItemSet((seeds, labels), names=("seeds", "labels"))
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    seeds_ids = []
    labels = []
    for i, minibatch in enumerate(item_sampler):
        assert minibatch.seeds is not None
        assert isinstance(minibatch.seeds, torch.Tensor)
        assert minibatch.labels is not None
        label = minibatch.labels
        assert len(minibatch.seeds) == len(label)
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = num_ids % batch_size
            else:
                assert False
        assert minibatch.seeds.shape == (expected_batch_size, 3)
        assert len(label) == expected_batch_size
        # Verify seeds and labels match.
        assert torch.equal(minibatch.seeds[:, 0] + 1, minibatch.seeds[:, 1])
        assert torch.equal(minibatch.seeds[:, 1] + 1, minibatch.seeds[:, 2])
        # Archive batch.
        seeds_ids.append(minibatch.seeds)
        labels.append(label)
    seeds_ids = torch.cat(seeds_ids)
    labels = torch.cat(labels)
    assert torch.all(seeds_ids[:-1, 0] <= seeds_ids[1:, 0]) is not shuffle
    assert torch.all(seeds_ids[:-1, 1] <= seeds_ids[1:, 1]) is not shuffle
    assert torch.all(seeds_ids[:-1, 2] <= seeds_ids[1:, 2]) is not shuffle
    assert torch.all(labels[:-1] <= labels[1:]) is not shuffle


def test_append_with_other_datapipes():
    num_ids = 100
    batch_size = 4
    item_set = gb.ItemSet(torch.arange(0, num_ids), names="seed_nodes")
    data_pipe = gb.ItemSampler(item_set, batch_size)
    # torchdata.datapipes.iter.Enumerator
    data_pipe = data_pipe.enumerate()
    for i, (idx, data) in enumerate(data_pipe):
        assert i == idx
        assert len(data.seeds) == batch_size


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSetDict_iterable_only(batch_size, shuffle, drop_last):
    class IterableOnly:
        def __init__(self, start, stop):
            self._start = start
            self._stop = stop

        def __iter__(self):
            return iter(torch.arange(self._start, self._stop))

    num_ids = 205
    ids = {
        "user": gb.ItemSet(IterableOnly(0, 99), names="seed_nodes"),
        "item": gb.ItemSet(IterableOnly(99, num_ids), names="seed_nodes"),
    }
    chained_ids = []
    for key, value in ids.items():
        chained_ids += [(key, v) for v in value]
    item_set = gb.ItemSetDict(ids)
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    minibatch_ids = []
    for i, minibatch in enumerate(item_sampler):
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = num_ids % batch_size
            else:
                assert False
        assert isinstance(minibatch, gb.MiniBatch)
        assert minibatch.seeds is not None
        ids = []
        for _, v in minibatch.seeds.items():
            ids.append(v)
        ids = torch.cat(ids)
        assert len(ids) == expected_batch_size
        minibatch_ids.append(ids)
    minibatch_ids = torch.cat(minibatch_ids)
    assert torch.all(minibatch_ids[:-1] <= minibatch_ids[1:]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSetDict_seed_nodes(batch_size, shuffle, drop_last):
    # Node IDs.
    num_ids = 205
    ids = {
        "user": gb.ItemSet(torch.arange(0, 99), names="seed_nodes"),
        "item": gb.ItemSet(torch.arange(99, num_ids), names="seed_nodes"),
    }
    chained_ids = []
    for key, value in ids.items():
        chained_ids += [(key, v) for v in value]
    item_set = gb.ItemSetDict(ids)
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    minibatch_ids = []
    for i, minibatch in enumerate(item_sampler):
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = num_ids % batch_size
            else:
                assert False
        assert isinstance(minibatch, gb.MiniBatch)
        assert minibatch.seeds is not None
        ids = []
        for _, v in minibatch.seeds.items():
            ids.append(v)
        ids = torch.cat(ids)
        assert len(ids) == expected_batch_size
        minibatch_ids.append(ids)
    minibatch_ids = torch.cat(minibatch_ids)
    assert torch.all(minibatch_ids[:-1] <= minibatch_ids[1:]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSetDict_seed_nodes_labels(batch_size, shuffle, drop_last):
    # Node IDs.
    num_ids = 205
    ids = {
        "user": gb.ItemSet(
            (torch.arange(0, 99), torch.arange(0, 99)),
            names=("seed_nodes", "labels"),
        ),
        "item": gb.ItemSet(
            (torch.arange(99, num_ids), torch.arange(99, num_ids)),
            names=("seed_nodes", "labels"),
        ),
    }
    chained_ids = []
    for key, value in ids.items():
        chained_ids += [(key, v) for v in value]
    item_set = gb.ItemSetDict(ids)
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    minibatch_ids = []
    minibatch_labels = []
    for i, minibatch in enumerate(item_sampler):
        assert isinstance(minibatch, gb.MiniBatch)
        assert minibatch.seeds is not None
        assert minibatch.labels is not None
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = num_ids % batch_size
            else:
                assert False
        ids = []
        for _, v in minibatch.seeds.items():
            ids.append(v)
        ids = torch.cat(ids)
        assert len(ids) == expected_batch_size
        minibatch_ids.append(ids)
        labels = []
        for _, v in minibatch.labels.items():
            labels.append(v)
        labels = torch.cat(labels)
        assert len(labels) == expected_batch_size
        minibatch_labels.append(labels)
    minibatch_ids = torch.cat(minibatch_ids)
    minibatch_labels = torch.cat(minibatch_labels)
    assert torch.all(minibatch_ids[:-1] <= minibatch_ids[1:]) is not shuffle
    assert (
        torch.all(minibatch_labels[:-1] <= minibatch_labels[1:]) is not shuffle
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSetDict_node_pairs(batch_size, shuffle, drop_last):
    # Node pairs.
    num_ids = 103
    total_pairs = 2 * num_ids
    node_pairs_like = torch.arange(0, num_ids * 2).reshape(-1, 2)
    node_pairs_follow = torch.arange(num_ids * 2, num_ids * 4).reshape(-1, 2)
    node_pairs_dict = {
        "user:like:item": gb.ItemSet(node_pairs_like, names="node_pairs"),
        "user:follow:user": gb.ItemSet(node_pairs_follow, names="node_pairs"),
    }
    item_set = gb.ItemSetDict(node_pairs_dict)
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    src_ids = []
    dst_ids = []
    for i, minibatch in enumerate(item_sampler):
        assert isinstance(minibatch, gb.MiniBatch)
        assert minibatch.seeds is not None
        assert minibatch.labels is None
        is_last = (i + 1) * batch_size >= total_pairs
        if not is_last or total_pairs % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = total_pairs % batch_size
            else:
                assert False
        src = []
        dst = []
        for _, (seeds) in minibatch.seeds.items():
            assert isinstance(seeds, torch.Tensor)
            src.append(seeds[:, 0])
            dst.append(seeds[:, 1])
        src = torch.cat(src)
        dst = torch.cat(dst)
        assert len(src) == expected_batch_size
        assert len(dst) == expected_batch_size
        src_ids.append(src)
        dst_ids.append(dst)
        assert torch.equal(src + 1, dst)
    src_ids = torch.cat(src_ids)
    dst_ids = torch.cat(dst_ids)
    assert torch.all(src_ids[:-1] <= src_ids[1:]) is not shuffle
    assert torch.all(dst_ids[:-1] <= dst_ids[1:]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSetDict_node_pairs_labels(batch_size, shuffle, drop_last):
    # Node pairs and labels
    num_ids = 103
    total_ids = 2 * num_ids
    node_pairs_like = torch.arange(0, num_ids * 2).reshape(-1, 2)
    node_pairs_follow = torch.arange(num_ids * 2, num_ids * 4).reshape(-1, 2)
    labels = torch.arange(0, num_ids)
    node_pairs_dict = {
        "user:like:item": gb.ItemSet(
            (node_pairs_like, node_pairs_like[:, 0]),
            names=("node_pairs", "labels"),
        ),
        "user:follow:user": gb.ItemSet(
            (node_pairs_follow, node_pairs_follow[:, 0]),
            names=("node_pairs", "labels"),
        ),
    }
    item_set = gb.ItemSetDict(node_pairs_dict)
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    src_ids = []
    dst_ids = []
    labels = []
    for i, minibatch in enumerate(item_sampler):
        assert isinstance(minibatch, gb.MiniBatch)
        assert minibatch.seeds is not None
        assert minibatch.labels is not None
        assert minibatch.negative_dsts is None
        is_last = (i + 1) * batch_size >= total_ids
        if not is_last or total_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = total_ids % batch_size
            else:
                assert False
        src = []
        dst = []
        label = []
        for _, seeds in minibatch.seeds.items():
            assert isinstance(seeds, torch.Tensor)
            src.append(seeds[:, 0])
            dst.append(seeds[:, 1])
        for _, v_label in minibatch.labels.items():
            label.append(v_label)
        src = torch.cat(src)
        dst = torch.cat(dst)
        label = torch.cat(label)
        assert len(src) == expected_batch_size
        assert len(dst) == expected_batch_size
        assert len(label) == expected_batch_size
        src_ids.append(src)
        dst_ids.append(dst)
        labels.append(label)
        assert torch.equal(src + 1, dst)
        assert torch.equal(src, label)
    src_ids = torch.cat(src_ids)
    dst_ids = torch.cat(dst_ids)
    labels = torch.cat(labels)
    assert torch.all(src_ids[:-1] <= src_ids[1:]) is not shuffle
    assert torch.all(dst_ids[:-1] <= dst_ids[1:]) is not shuffle
    assert torch.all(labels[:-1] <= labels[1:]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSetDict_node_pairs_negative_dsts(batch_size, shuffle, drop_last):
    # Head, tail and negative tails.
    num_ids = 103
    total_ids = 2 * num_ids
    num_negs = 2
    node_paris_like = torch.arange(0, num_ids * 2).reshape(-1, 2)
    node_pairs_follow = torch.arange(num_ids * 2, num_ids * 4).reshape(-1, 2)
    neg_dsts_like = torch.arange(
        num_ids * 4, num_ids * 4 + num_ids * num_negs
    ).reshape(-1, num_negs)
    neg_dsts_follow = torch.arange(
        num_ids * 4 + num_ids * num_negs, num_ids * 4 + num_ids * num_negs * 2
    ).reshape(-1, num_negs)
    data_dict = {
        "user:like:item": gb.ItemSet(
            (node_paris_like, neg_dsts_like),
            names=("node_pairs", "negative_dsts"),
        ),
        "user:follow:user": gb.ItemSet(
            (node_pairs_follow, neg_dsts_follow),
            names=("node_pairs", "negative_dsts"),
        ),
    }
    item_set = gb.ItemSetDict(data_dict)
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    src_ids = []
    dst_ids = []
    negs_ids = []
    for i, minibatch in enumerate(item_sampler):
        assert isinstance(minibatch, gb.MiniBatch)
        assert minibatch.seeds is not None
        assert minibatch.labels is not None
        assert minibatch.negative_dsts is None
        is_last = (i + 1) * batch_size >= total_ids
        if not is_last or total_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = total_ids % batch_size
            else:
                assert False
        src = []
        dst = []
        negs_src = []
        negs_dst = []
        for etype, seeds in minibatch.seeds.items():
            assert isinstance(seeds, torch.Tensor)
            src_etype = seeds[:, 0]
            dst_etype = seeds[:, 1]
            src.append(src_etype[minibatch.labels[etype].to(bool)])
            dst.append(dst_etype[minibatch.labels[etype].to(bool)])
            negs_src.append(src_etype[~minibatch.labels[etype].to(bool)])
            negs_dst.append(dst_etype[~minibatch.labels[etype].to(bool)])
        src = torch.cat(src)
        dst = torch.cat(dst)
        negs_src = torch.cat(negs_src)
        negs_dst = torch.cat(negs_dst)
        assert len(src) == expected_batch_size
        assert len(dst) == expected_batch_size
        assert len(negs_src) == expected_batch_size * 2
        assert len(negs_dst) == expected_batch_size * 2
        src_ids.append(src)
        dst_ids.append(dst)
        negs_ids.append(negs_dst)
        assert negs_src.dim() == 1
        assert negs_dst.dim() == 1
        assert torch.equal(src + 1, dst)
        assert torch.equal(negs_src, (negs_dst - num_ids * 4) // 2 * 2)
    src_ids = torch.cat(src_ids)
    dst_ids = torch.cat(dst_ids)
    negs_ids = torch.cat(negs_ids)
    assert torch.all(src_ids[:-1] <= src_ids[1:]) is not shuffle
    assert torch.all(dst_ids[:-1] <= dst_ids[1:]) is not shuffle
    assert torch.all(negs_ids <= negs_ids) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSetDict_seeds(batch_size, shuffle, drop_last):
    # Node pairs.
    num_ids = 103
    total_pairs = 2 * num_ids
    seeds_like = torch.arange(0, num_ids * 3).reshape(-1, 3)
    seeds_follow = torch.arange(num_ids * 3, num_ids * 6).reshape(-1, 3)
    seeds_dict = {
        "user:like:item": gb.ItemSet(seeds_like, names="seeds"),
        "user:follow:user": gb.ItemSet(seeds_follow, names="seeds"),
    }
    item_set = gb.ItemSetDict(seeds_dict)
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    seeds_ids = []
    for i, minibatch in enumerate(item_sampler):
        assert isinstance(minibatch, gb.MiniBatch)
        assert minibatch.seeds is not None
        assert minibatch.labels is None
        assert minibatch.indexes is None
        is_last = (i + 1) * batch_size >= total_pairs
        if not is_last or total_pairs % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = total_pairs % batch_size
            else:
                assert False
        seeds_lst = []
        for _, (seeds) in minibatch.seeds.items():
            assert isinstance(seeds, torch.Tensor)
            seeds_lst.append(seeds)
        seeds_lst = torch.cat(seeds_lst)
        assert seeds_lst.shape == (expected_batch_size, 3)
        seeds_ids.append(seeds_lst)
        assert torch.equal(seeds_lst[:, 0] + 1, seeds_lst[:, 1])
        assert torch.equal(seeds_lst[:, 1] + 1, seeds_lst[:, 2])
    seeds_ids = torch.cat(seeds_ids)
    assert torch.all(seeds_ids[:-1, 0] <= seeds_ids[1:, 0]) is not shuffle
    assert torch.all(seeds_ids[:-1, 1] <= seeds_ids[1:, 1]) is not shuffle
    assert torch.all(seeds_ids[:-1, 2] <= seeds_ids[1:, 2]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSetDict_seeds_labels(batch_size, shuffle, drop_last):
    # Node pairs and labels
    num_ids = 103
    total_ids = 2 * num_ids
    seeds_like = torch.arange(0, num_ids * 3).reshape(-1, 3)
    seeds_follow = torch.arange(num_ids * 3, num_ids * 6).reshape(-1, 3)
    seeds_dict = {
        "user:like:item": gb.ItemSet(
            (seeds_like, seeds_like[:, 0]),
            names=("seeds", "labels"),
        ),
        "user:follow:user": gb.ItemSet(
            (seeds_follow, seeds_follow[:, 0]),
            names=("seeds", "labels"),
        ),
    }
    item_set = gb.ItemSetDict(seeds_dict)
    item_sampler = gb.ItemSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    seeds_ids = []
    labels = []
    for i, minibatch in enumerate(item_sampler):
        assert isinstance(minibatch, gb.MiniBatch)
        assert minibatch.seeds is not None
        assert minibatch.labels is not None
        assert minibatch.indexes is None
        is_last = (i + 1) * batch_size >= total_ids
        if not is_last or total_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = total_ids % batch_size
            else:
                assert False
        seeds_lst = []
        label = []
        for _, seeds in minibatch.seeds.items():
            assert isinstance(seeds, torch.Tensor)
            seeds_lst.append(seeds)
        for _, v_label in minibatch.labels.items():
            label.append(v_label)
        seeds_lst = torch.cat(seeds_lst)
        label = torch.cat(label)
        assert seeds_lst.shape == (expected_batch_size, 3)
        assert len(label) == expected_batch_size
        seeds_ids.append(seeds_lst)
        labels.append(label)
        assert torch.equal(seeds_lst[:, 0] + 1, seeds_lst[:, 1])
        assert torch.equal(seeds_lst[:, 1] + 1, seeds_lst[:, 2])
        assert torch.equal(seeds_lst[:, 0], label)
    seeds_ids = torch.cat(seeds_ids)
    labels = torch.cat(labels)
    assert torch.all(seeds_ids[:-1, 0] <= seeds_ids[1:, 0]) is not shuffle
    assert torch.all(seeds_ids[:-1, 1] <= seeds_ids[1:, 1]) is not shuffle
    assert torch.all(seeds_ids[:-1, 2] <= seeds_ids[1:, 2]) is not shuffle
    assert torch.all(labels[:-1] <= labels[1:]) is not shuffle


def distributed_item_sampler_subprocess(
    proc_id,
    nprocs,
    item_set,
    num_ids,
    num_workers,
    batch_size,
    drop_last,
    drop_uneven_inputs,
):
    # On Windows, the init method can only be file.
    init_method = (
        f"file:///{os.path.join(os.getcwd(), 'dis_tempfile')}"
        if platform == "win32"
        else "tcp://127.0.0.1:12345"
    )
    dist.init_process_group(
        backend="gloo",  # Use Gloo backend for CPU multiprocessing
        init_method=init_method,
        world_size=nprocs,
        rank=proc_id,
    )

    # Create a DistributedItemSampler.
    item_sampler = gb.DistributedItemSampler(
        item_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        drop_uneven_inputs=drop_uneven_inputs,
    )
    feature_fetcher = gb.FeatureFetcher(
        item_sampler,
        gb.BasicFeatureStore({}),
        [],
    )
    data_loader = gb.DataLoader(feature_fetcher, num_workers=num_workers)

    # Count the numbers of items and batches.
    num_items = 0
    sampled_count = torch.zeros(num_ids, dtype=torch.int32)
    for i in data_loader:
        # Count how many times each item is sampled.
        sampled_count[i.seeds] += 1
        if drop_last:
            assert i.seeds.size(0) == batch_size
        num_items += i.seeds.size(0)
    num_batches = len(list(item_sampler))

    if drop_uneven_inputs:
        num_batches_tensor = torch.tensor(num_batches)
        dist.broadcast(num_batches_tensor, 0)
        # Test if the number of batches are the same for all processes.
        assert num_batches_tensor == num_batches

    # Add up results from all processes.
    dist.reduce(sampled_count, 0)

    try:
        # Make sure no item is sampled more than once.
        assert sampled_count.max() <= 1
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "params",
    [
        ((24, 4, 0, 4, False, False), [(8, 8), (8, 8), (4, 4), (4, 4)]),
        ((30, 4, 0, 4, False, False), [(8, 8), (8, 8), (8, 8), (6, 6)]),
        ((30, 4, 0, 4, True, False), [(8, 8), (8, 8), (8, 8), (6, 4)]),
        ((30, 4, 0, 4, False, True), [(8, 8), (8, 8), (8, 8), (6, 6)]),
        ((30, 4, 0, 4, True, True), [(8, 4), (8, 4), (8, 4), (6, 4)]),
        (
            (53, 4, 2, 4, False, False),
            [(8, 8), (8, 8), (8, 8), (5, 5), (8, 8), (4, 4), (8, 8), (4, 4)],
        ),
        (
            (53, 4, 2, 4, True, False),
            [(8, 8), (8, 8), (9, 8), (4, 4), (8, 8), (4, 4), (8, 8), (4, 4)],
        ),
        (
            (53, 4, 2, 4, False, True),
            [(10, 8), (6, 4), (9, 8), (4, 4), (8, 8), (4, 4), (8, 8), (4, 4)],
        ),
        (
            (53, 4, 2, 4, True, True),
            [(10, 8), (6, 4), (9, 8), (4, 4), (8, 8), (4, 4), (8, 8), (4, 4)],
        ),
        (
            (63, 4, 2, 4, False, False),
            [(8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (7, 7)],
        ),
        (
            (63, 4, 2, 4, True, False),
            [(8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (10, 8), (5, 4)],
        ),
        (
            (63, 4, 2, 4, False, True),
            [(8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (7, 7)],
        ),
        (
            (63, 4, 2, 4, True, True),
            [
                (10, 8),
                (6, 4),
                (10, 8),
                (6, 4),
                (10, 8),
                (6, 4),
                (10, 8),
                (5, 4),
            ],
        ),
        (
            (65, 4, 2, 4, False, False),
            [(9, 9), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8)],
        ),
        (
            (65, 4, 2, 4, True, True),
            [(9, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8)],
        ),
    ],
)
def test_RangeCalculation(params):
    (
        (
            total,
            num_replicas,
            num_workers,
            batch_size,
            drop_last,
            drop_uneven_inputs,
        ),
        key,
    ) = params
    answer = []
    sum = 0
    for rank in range(num_replicas):
        for worker_id in range(max(num_workers, 1)):
            result = gb.internal.calculate_range(
                True,
                total,
                num_replicas,
                rank,
                num_workers,
                worker_id,
                batch_size,
                drop_last,
                drop_uneven_inputs,
            )
            assert sum == result[0]
            sum += result[1]
            answer.append((result[1], result[2]))
    assert key == answer


@pytest.mark.parametrize("num_ids", [24, 30, 32, 34, 36])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("drop_uneven_inputs", [False, True])
def test_DistributedItemSampler(
    num_ids, num_workers, drop_last, drop_uneven_inputs
):
    nprocs = 4
    batch_size = 4
    item_set = gb.ItemSet(torch.arange(0, num_ids), names="seed_nodes")

    # On Windows, if the process group initialization file already exists,
    # the program may hang. So we need to delete it if it exists.
    if platform == "win32":
        try:
            os.remove(os.path.join(os.getcwd(), "dis_tempfile"))
        except FileNotFoundError:
            pass

    mp.spawn(
        distributed_item_sampler_subprocess,
        args=(
            nprocs,
            item_set,
            num_ids,
            num_workers,
            batch_size,
            drop_last,
            drop_uneven_inputs,
        ),
        nprocs=nprocs,
        join=True,
    )
