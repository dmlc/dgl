import dgl
import pytest
import torch
from dgl import graphbolt as gb
from torch.testing import assert_close


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSet_node_ids(batch_size, shuffle, drop_last):
    # Node IDs.
    num_ids = 103
    item_set = gb.ItemSet(torch.arange(0, num_ids))
    minibatch_sampler = gb.MinibatchSampler(
        item_set,
        batch_size=batch_size,
        data_block_mapper=gb.node_classification_mapper,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    minibatch_ids = []
    for i, minibatch in enumerate(minibatch_sampler):
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            assert len(minibatch.seed_node) == batch_size
        else:
            if not drop_last:
                assert len(minibatch.seed_node) == num_ids % batch_size
            else:
                assert False
        minibatch_ids.append(minibatch.seed_node)
    minibatch_ids = torch.cat(minibatch_ids)
    assert torch.all(minibatch_ids[:-1] <= minibatch_ids[1:]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSet_node_ids_labels(batch_size, shuffle, drop_last):
    # Node IDs and labels.
    num_ids = 103
    item_set = gb.ItemSet((torch.arange(0, num_ids), torch.arange(0, num_ids)))
    minibatch_sampler = gb.MinibatchSampler(
        item_set,
        batch_size=batch_size,
        data_block_mapper=gb.node_classification_mapper,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    minibatch_ids = []
    minibatch_labels = []
    for i, minibatch in enumerate(minibatch_sampler):
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            assert len(minibatch.seed_node) == batch_size
            assert len(minibatch.label) == batch_size
        else:
            if not drop_last:
                assert len(minibatch.seed_node) == num_ids % batch_size
                assert len(minibatch.label) == num_ids % batch_size
            else:
                assert False
        minibatch_ids.append(minibatch.seed_node)
        minibatch_labels.append(minibatch.label)
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
    item_set = gb.ItemSet(graphs)
    minibatch_sampler = gb.MinibatchSampler(
        item_set,
        batch_size=batch_size,
        data_block_mapper=gb.graph_classification_mapper,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    minibatch_num_nodes = []
    minibatch_num_edges = []
    for i, minibatch in enumerate(minibatch_sampler):
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
    node_pairs = (torch.arange(0, num_ids), torch.arange(num_ids, num_ids * 2))
    item_set = gb.ItemSet(node_pairs)
    minibatch_sampler = gb.MinibatchSampler(
        item_set,
        batch_size=batch_size,
        data_block_mapper=gb.link_prediction_mapper,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    src_ids = []
    dst_ids = []
    for i, minibatch in enumerate(minibatch_sampler):
        src, dst = minibatch.node_pair
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
        assert torch.equal(src + num_ids, dst)
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
    node_pairs = (torch.arange(0, num_ids), torch.arange(num_ids, num_ids * 2))
    labels = torch.arange(0, num_ids)
    item_set = gb.ItemSet((node_pairs[0], node_pairs[1], labels))
    minibatch_sampler = gb.MinibatchSampler(
        item_set,
        batch_size=batch_size,
        data_block_mapper=gb.link_prediction_mapper,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    src_ids = []
    dst_ids = []
    labels = []
    for i, minibatch in enumerate(minibatch_sampler):
        src, dst = minibatch.node_pair
        label = minibatch.label
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
        assert torch.equal(src + num_ids, dst)
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
def test_ItemSet_head_tail_neg_tails_labels(batch_size, shuffle, drop_last):
    # Head, tail and negative tails.
    num_ids = 103
    num_negs = 2
    heads = torch.arange(0, num_ids)
    tails = torch.arange(num_ids, num_ids * 2)
    neg_tails = torch.stack((heads + 1, heads + 2), dim=-1)
    labels = torch.arange(0, num_ids)
    item_set = gb.ItemSet((heads, tails, neg_tails, labels))
    for i, (head, tail, negs, label) in enumerate(item_set):
        assert heads[i] == head
        assert tails[i] == tail
        assert torch.equal(neg_tails[i], negs)
        assert labels[i] == label
    minibatch_sampler = gb.MinibatchSampler(
        item_set,
        batch_size=batch_size,
        data_block_mapper=gb.link_prediction_mapper,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    head_ids = []
    tail_ids = []
    negs_ids = []
    for i, minibatch in enumerate(minibatch_sampler):
        head, tail = minibatch.node_pair
        negs = minibatch.negative_tail
        label = minibatch.label
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = num_ids % batch_size
            else:
                assert False
        assert len(head) == expected_batch_size
        assert len(tail) == expected_batch_size
        assert negs.dim() == 2
        assert negs.shape[0] == expected_batch_size
        assert negs.shape[1] == num_negs
        # Verify head/tail and negatie tails match.
        assert torch.equal(head + num_ids, tail)
        assert torch.equal(head + 1, negs[:, 0])
        assert torch.equal(head + 2, negs[:, 1])
        # Archive batch.
        head_ids.append(head)
        tail_ids.append(tail)
        negs_ids.append(negs)
    head_ids = torch.cat(head_ids)
    tail_ids = torch.cat(tail_ids)
    negs_ids = torch.cat(negs_ids)
    assert torch.all(head_ids[:-1] <= head_ids[1:]) is not shuffle
    assert torch.all(tail_ids[:-1] <= tail_ids[1:]) is not shuffle
    assert torch.all(negs_ids[:-1, 0] <= negs_ids[1:, 0]) is not shuffle
    assert torch.all(negs_ids[:-1, 1] <= negs_ids[1:, 1]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSet_head_tail_neg_head_tails_labels(
    batch_size, shuffle, drop_last
):
    # Head, tail and negative tails.
    num_ids = 103
    num_negs = 2
    heads = torch.arange(0, num_ids)
    tails = torch.arange(num_ids, num_ids * 2)
    neg_heads = torch.stack((heads + 1, heads + 2), dim=-1)
    neg_tails = torch.stack((heads + 2, heads + 3), dim=-1)
    labels = torch.arange(0, num_ids)
    item_set = gb.ItemSet((heads, tails, neg_heads, neg_tails, labels))
    for i, (head, tail, nheads, ntails, label) in enumerate(item_set):
        assert heads[i] == head
        assert tails[i] == tail
        assert torch.equal(neg_heads[i], nheads)
        assert torch.equal(neg_tails[i], ntails)
        assert labels[i] == label
    minibatch_sampler = gb.MinibatchSampler(
        item_set,
        batch_size=batch_size,
        data_block_mapper=gb.link_prediction_mapper,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    head_ids = []
    tail_ids = []
    neg_heads_ids = []
    neg_tails_ids = []
    for i, minibatch in enumerate(minibatch_sampler):
        head, tail = minibatch.node_pair
        neg_heads = minibatch.negative_head
        neg_tails = minibatch.negative_tail
        label = minibatch.label
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = num_ids % batch_size
            else:
                assert False
        assert len(head) == expected_batch_size
        assert len(tail) == expected_batch_size
        assert neg_heads.dim() == 2
        assert neg_heads.shape[0] == expected_batch_size
        assert neg_heads.shape[1] == num_negs
        assert neg_tails.dim() == 2
        assert neg_tails.shape[0] == expected_batch_size
        assert neg_tails.shape[1] == num_negs
        # Verify head/tail and negatie heads/tails match.
        assert torch.equal(head + num_ids, tail)
        assert torch.equal(head + 1, neg_heads[:, 0])
        assert torch.equal(head + 2, neg_heads[:, 1])
        assert torch.equal(head + 2, neg_tails[:, 0])
        assert torch.equal(head + 3, neg_tails[:, 1])
        # Archive batch.
        head_ids.append(head)
        tail_ids.append(tail)
        neg_heads_ids.append(neg_heads)
        neg_tails_ids.append(neg_tails)
    head_ids = torch.cat(head_ids)
    tail_ids = torch.cat(tail_ids)
    neg_heads_ids = torch.cat(neg_heads_ids)
    neg_tails_ids = torch.cat(neg_tails_ids)
    assert torch.all(head_ids[:-1] <= head_ids[1:]) is not shuffle
    assert torch.all(tail_ids[:-1] <= tail_ids[1:]) is not shuffle
    assert (
        torch.all(neg_heads_ids[:-1, 0] <= neg_heads_ids[1:, 0]) is not shuffle
    )
    assert (
        torch.all(neg_heads_ids[:-1, 1] <= neg_heads_ids[1:, 1]) is not shuffle
    )
    assert (
        torch.all(neg_tails_ids[:-1, 0] <= neg_tails_ids[1:, 0]) is not shuffle
    )
    assert (
        torch.all(neg_tails_ids[:-1, 1] <= neg_tails_ids[1:, 1]) is not shuffle
    )


def test_append_with_other_datapipes():
    num_ids = 100
    batch_size = 4
    item_set = gb.ItemSet(torch.arange(0, num_ids))
    data_pipe = gb.MinibatchSampler(item_set, batch_size)
    # torchdata.datapipes.iter.Enumerator
    data_pipe = data_pipe.enumerate()
    for i, (idx, data) in enumerate(data_pipe):
        assert i == idx
        assert len(data) == batch_size


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSetDict_node_ids(batch_size, shuffle, drop_last):
    # Node IDs.
    num_ids = 205
    ids = {
        "user": gb.ItemSet(torch.arange(0, 99)),
        "item": gb.ItemSet(torch.arange(99, num_ids)),
    }
    chained_ids = []
    for key, value in ids.items():
        chained_ids += [(key, v) for v in value]
    item_set = gb.ItemSetDict(ids)
    minibatch_sampler = gb.MinibatchSampler(
        item_set,
        batch_size=batch_size,
        data_block_mapper=gb.node_classification_mapper,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    minibatch_ids = []
    for i, minibatch in enumerate(minibatch_sampler):
        batch = minibatch.seed_node
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = num_ids % batch_size
            else:
                assert False
        assert isinstance(batch, dict)
        ids = []
        for _, v in batch.items():
            ids.append(v)
        ids = torch.cat(ids)
        assert len(ids) == expected_batch_size
        minibatch_ids.append(ids)
    minibatch_ids = torch.cat(minibatch_ids)
    assert torch.all(minibatch_ids[:-1] <= minibatch_ids[1:]) is not shuffle


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ItemSetDict_node_ids_labels(batch_size, shuffle, drop_last):
    # Node IDs.
    num_ids = 205
    ids = {
        "user": gb.ItemSet((torch.arange(0, 99), torch.arange(0, 99))),
        "item": gb.ItemSet(
            (torch.arange(99, num_ids), torch.arange(99, num_ids))
        ),
    }
    chained_ids = []
    for key, value in ids.items():
        chained_ids += [(key, v) for v in value]
    item_set = gb.ItemSetDict(ids)
    minibatch_sampler = gb.MinibatchSampler(
        item_set,
        batch_size=batch_size,
        data_block_mapper=gb.node_classification_mapper,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    minibatch_ids = []
    minibatch_labels = []
    for i, minibatch in enumerate(minibatch_sampler):
        batch = minibatch.seed_node
        label = minibatch.label
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = num_ids % batch_size
            else:
                assert False
        assert isinstance(batch, dict)
        assert isinstance(label, dict)
        ids = []
        for _, v in batch.items():
            ids.append(v)
        ids = torch.cat(ids)
        assert len(ids) == expected_batch_size
        minibatch_ids.append(ids)
        labels = []
        for _, v in label.items():
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
    total_ids = 2 * num_ids
    node_pairs_0 = (
        torch.arange(0, num_ids),
        torch.arange(num_ids, num_ids * 2),
    )
    node_pairs_1 = (
        torch.arange(num_ids * 2, num_ids * 3),
        torch.arange(num_ids * 3, num_ids * 4),
    )
    node_pairs_dict = {
        "user:like:item": gb.ItemSet(node_pairs_0),
        "user:follow:user": gb.ItemSet(node_pairs_1),
    }
    item_set = gb.ItemSetDict(node_pairs_dict)
    minibatch_sampler = gb.MinibatchSampler(
        item_set,
        batch_size=batch_size,
        data_block_mapper=gb.link_prediction_mapper,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    src_ids = []
    dst_ids = []
    for i, minibatch in enumerate(minibatch_sampler):
        batch = minibatch.node_pair
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
        for _, (v_src, v_dst) in batch.items():
            src.append(v_src)
            dst.append(v_dst)
        src = torch.cat(src)
        dst = torch.cat(dst)
        assert len(src) == expected_batch_size
        assert len(dst) == expected_batch_size
        src_ids.append(src)
        dst_ids.append(dst)
        assert torch.equal(src + num_ids, dst)
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
    node_pairs_0 = (
        torch.arange(0, num_ids),
        torch.arange(num_ids, num_ids * 2),
    )
    node_pairs_1 = (
        torch.arange(num_ids * 2, num_ids * 3),
        torch.arange(num_ids * 3, num_ids * 4),
    )
    labels = torch.arange(0, num_ids)
    node_pairs_dict = {
        "user:like:item": gb.ItemSet(
            (node_pairs_0[0], node_pairs_0[1], labels)
        ),
        "user:follow:user": gb.ItemSet(
            (node_pairs_1[0], node_pairs_1[1], labels + num_ids * 2)
        ),
    }
    item_set = gb.ItemSetDict(node_pairs_dict)
    minibatch_sampler = gb.MinibatchSampler(
        item_set,
        batch_size=batch_size,
        data_block_mapper=gb.link_prediction_mapper,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    src_ids = []
    dst_ids = []
    labels = []
    for i, batch in enumerate(minibatch_sampler):
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
        for _, (v_src, v_dst) in batch.node_pair.items():
            src.append(v_src)
            dst.append(v_dst)
        for _, v_label in batch.label.items():
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
        assert torch.equal(src + num_ids, dst)
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
def test_ItemSetDict_head_tail_neg_tails_labels(batch_size, shuffle, drop_last):
    # Head, tail, negative tails and labels.
    num_ids = 103
    total_ids = 2 * num_ids
    num_negs = 2
    heads = torch.arange(0, num_ids)
    tails = torch.arange(num_ids, num_ids * 2)
    neg_tails = torch.stack((heads + 1, heads + 2), dim=-1)
    labels = torch.arange(0, num_ids)
    data_dict = {
        "user:like:item": gb.ItemSet((heads, tails, neg_tails, labels)),
        "user:follow:user": gb.ItemSet(
            (heads, tails, neg_tails, labels + num_ids * 2)
        ),
    }
    item_set = gb.ItemSetDict(data_dict)
    minibatch_sampler = gb.MinibatchSampler(
        item_set,
        batch_size=batch_size,
        data_block_mapper=gb.link_prediction_mapper,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    head_ids = []
    tail_ids = []
    negs_ids = []
    label_ids = []
    for i, batch in enumerate(minibatch_sampler):
        is_last = (i + 1) * batch_size >= total_ids
        if not is_last or total_ids % batch_size == 0:
            expected_batch_size = batch_size
        else:
            if not drop_last:
                expected_batch_size = total_ids % batch_size
            else:
                assert False
        head = []
        tail = []
        negs = []
        labels = []
        for _, (v_head, v_tail) in batch.node_pair.items():
            head.append(v_head)
            tail.append(v_tail)
        for _, v_negs in batch.negative_tail.items():
            negs.append(v_negs)
        for _, label in batch.label.items():
            labels.append(label)
        head = torch.cat(head)
        tail = torch.cat(tail)
        negs = torch.cat(negs)
        labels = torch.cat(labels)
        assert len(head) == expected_batch_size
        assert len(tail) == expected_batch_size
        assert len(negs) == expected_batch_size
        assert len(labels) == expected_batch_size
        head_ids.append(head)
        tail_ids.append(tail)
        negs_ids.append(negs)
        label_ids.append(labels)
        assert negs.dim() == 2
        assert negs.shape[0] == expected_batch_size
        assert negs.shape[1] == num_negs
        assert torch.equal(head + num_ids, tail)
        assert torch.equal(head + 1, negs[:, 0])
        assert torch.equal(head + 2, negs[:, 1])
    head_ids = torch.cat(head_ids)
    tail_ids = torch.cat(tail_ids)
    negs_ids = torch.cat(negs_ids)
    label_ids = torch.cat(label_ids)
    assert torch.all(head_ids[:-1] <= head_ids[1:]) is not shuffle
    assert torch.all(tail_ids[:-1] <= tail_ids[1:]) is not shuffle
    assert torch.all(negs_ids[:-1] <= negs_ids[1:]) is not shuffle
    assert torch.all(label_ids[:-1] <= label_ids[1:]) is not shuffle
